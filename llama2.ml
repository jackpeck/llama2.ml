#load "unix.cma";;







type args2 = {
  mutable checkpoint : string;
  mutable temperature : float;
  mutable steps : int;
  mutable prompt : string;
}

(* let run args =
  match args.checkpoint, args.temperature, args.steps, args.prompt with
  | checkpoint, temperature, steps, prompt ->
      Printf.printf
        "Checkpoint: %s, Temperature: %f, Steps: %d, Prompt: %s\n"
        checkpoint temperature steps prompt *)




let get_int32_le bytes offset =
  let b0 = Char.code (Bytes.get bytes (offset + 0)) in
  let b1 = Char.code (Bytes.get bytes (offset + 1)) in
  let b2 = Char.code (Bytes.get bytes (offset + 2)) in
  let b3 = Char.code (Bytes.get bytes (offset + 3)) in
  (b3 lsl 24) lor (b2 lsl 16) lor (b1 lsl 8) lor b0
;;


let print_bytes bytes =
  let len = Bytes.length bytes in
  print_string "b'";
  for i = 0 to len - 1 do
    Printf.printf "\\x%02x" (Char.code (Bytes.get bytes i));
  done;
  print_string "'";
  print_newline ()
;;

type config = {
  dim : int;
  hidden_dim : int;
  n_layers : int;
  n_heads : int;
  n_kv_heads : int;
  vocab_size : int;
  seq_len : int;
  shared_weights : int;
}

let create_config dim hidden_dim n_layers n_heads n_kv_heads vocab_size seq_len shared_weights =
  { dim; hidden_dim; n_layers; n_heads; n_kv_heads; vocab_size; seq_len; shared_weights }


let print_config config =
  Printf.printf
    "dim: %d, hidden_dim: %d, n_layers: %d, n_heads: %d, n_kv_heads: %d, vocab_size: %d, seq_len: %d, shared_weights: %d\n"
    config.dim config.hidden_dim config.n_layers config.n_heads config.n_kv_heads config.vocab_size config.seq_len config.shared_weights
    

let read_config file checkpoint =
  let size = 28 in (* bytes in int * 7 = 28 *)
  let config_header = Bytes.create size in
  try
    let read_bytes = input file config_header 0 size in
    (* close_in file; *)
    if read_bytes <> size then begin
      Printf.printf "Couldn't read config header from file %s\n" checkpoint;
      exit 1
    end;
    print_bytes config_header;

    let dim = get_int32_le config_header 0 in
    let hidden_dim = get_int32_le config_header 4 in
    let n_layers = get_int32_le config_header 8 in
    let n_heads = get_int32_le config_header 12 in
    let n_kv_heads = get_int32_le config_header 16 in
    let vocab_size = get_int32_le config_header 20 in
    let seq_len = get_int32_le config_header 24 in

    (* # negative vocab size is hacky way of signaling unshared weights. bit yikes. *)
    let shared_weights = if vocab_size > 0 then 1 else 0 in
    let config =
      create_config dim hidden_dim n_layers n_heads n_kv_heads (abs vocab_size) seq_len shared_weights in
    config
  with
  | Sys_error msg ->
    close_in file;
    Printf.printf "Couldn't open file %s: %s\n" checkpoint msg;
    exit 1
;;







type transformer_weights = {
  mutable token_embedding_table : float array;
  mutable rms_att_weight : float array;
  mutable wq : float array;
  mutable wk : float array;
  mutable wv : float array;
  mutable wo : float array;
  mutable rms_ffn_weight : float array;
  mutable w1 : float array;
  mutable w2 : float array;
  mutable w3 : float array;
  mutable rms_final_weight : float array;
  mutable freq_cis_real : float array;
  mutable freq_cis_imag : float array;
  mutable wcls : float array;
}


(* https://discuss.ocaml.org/t/pretty-printing-binary-ints/9062/7 *)
let int_size = Sys.word_size - 1
let int2bin =
  let buf = Bytes.create int_size in
  fun n ->
    for i = 0 to int_size - 1 do
      let pos = int_size - 1 - i in
      Bytes.set buf pos (if n land (1 lsl i) != 0 then '1' else '0')
    done;
    (* skip leading zeros *)
    (* match Bytes.index_opt buf '1' with
    | None -> "0b0"
    | Some i -> "0b" ^ Bytes.sub_string buf i (int_size - i) *)
    Bytes.sub_string buf 0 int_size

let int32_size = 32

let int32_to_bin n =
  let buf = Bytes.create int32_size in
  for i = 0 to int32_size - 1 do
    let pos = int32_size - 1 - i in
    Bytes.set buf pos (if Int32.logand n (Int32.shift_left (Int32.of_int 1) i) <> 0l then '1' else '0')
  done;
  Bytes.sub_string buf 0 int32_size


let float_decode bits =
  let p1 = Int32.shift_left (Int32.logand (Int32.shift_left(Int32.of_int 0b11111111) 0) bits) 24 in
  let p2 = Int32.shift_left (Int32.logand (Int32.shift_left(Int32.of_int 0b11111111) 8) bits) 8 in
  let p3 = Int32.shift_right (Int32.logand (Int32.shift_left(Int32.of_int 0b11111111) 16) bits) 8 in
  let p4 = Int32.shift_right (Int32.logand (Int32.shift_left(Int32.of_int 0b11111111) 24) bits) 24 in
  let sum = Int32.add (Int32.add p1 p2) (Int32.add p3 p4) in
  Int32.float_of_bits sum
;;


let input_binary_float file =
  let int_bits = input_binary_int file in
  float_decode (Int32.of_int int_bits)
;;


let checkpoint_init_weights weights conf file shared_weights file_size =
  let read_floats count =
    let values = Array.make count 0.0 in
    for i = 0 to count - 1 do
      values.(i) <- input_binary_float file;
    done;
    values
  in

  (* read_floats (31); *)
  weights.token_embedding_table <- read_floats (conf.vocab_size * conf.dim);
  weights.rms_att_weight <- read_floats (conf.n_layers * conf.dim);
  weights.wq <- read_floats (conf.n_layers * conf.dim * conf.dim);
  weights.wk <- read_floats (conf.n_layers * conf.dim * conf.dim);
  weights.wv <- read_floats (conf.n_layers * conf.dim * conf.dim);
  weights.wo <- read_floats (conf.n_layers * conf.dim * conf.dim);
  weights.rms_ffn_weight <- read_floats (conf.n_layers * conf.dim);
  weights.w1 <- read_floats (conf.n_layers * conf.dim * conf.hidden_dim);
  weights.w2 <- read_floats (conf.n_layers * conf.hidden_dim * conf.dim);
  weights.w3 <- read_floats (conf.n_layers * conf.dim * conf.hidden_dim);
  weights.rms_final_weight <- read_floats conf.dim;
  weights.freq_cis_real <- read_floats (conf.seq_len * (conf.dim / conf.n_heads) / 2);
  weights.freq_cis_imag <- read_floats (conf.seq_len * (conf.dim / conf.n_heads) / 2);

  (* The last line in Python code: *)
  if shared_weights = 1 then
    weights.wcls <- weights.token_embedding_table
  else
    weights.wcls <- read_floats ((file_size - pos_in file) / 4)



let create_transformer_weights () =
  {
    token_embedding_table = [||];
    rms_att_weight = [||];
    wq = [||];
    wk = [||];
    wv = [||];
    wo = [||];
    rms_ffn_weight = [||];
    w1 = [||];
    w2 = [||];
    w3 = [||];
    rms_final_weight = [||];
    freq_cis_real = [||];
    freq_cis_imag = [||];
    wcls = [||];
  }
    
;;
  

let print_token_embedding_table weights =
  Array.iteri (fun i value ->
    Printf.printf "token_embedding_table[%d]: %f\n" i value
  ) weights.token_embedding_table
;;

(* let input_binary typ file = match typ with
  | "f" -> input_binary_float file
  | "i" -> input_binary_int file *)

let tokenizer_init conf file =
  let vocab = ref [] in
  let vocab_scores = ref [] in
  let max_token_length = ref 0 in

  let read_float count =
    let values = Array.make count 0.0 in
    for i = 0 to count - 1 do
      values.(i) <- input_binary_float file;
    done;
    values
  in
  let read_int count =
    let values = Array.make count 0 in
    for i = 0 to count - 1 do
      values.(i) <- (Int.shift_right (input_binary_int file) 24);
    done;
    values
  in

  max_token_length := (read_int 1).(0);
  for _i = 0 to conf.vocab_size - 1 do
    vocab_scores := (read_float 1).(0) :: !vocab_scores;
    let len = (read_int 1).(0) in
    (* let bstr = Bytes.to_string (read_float len).(0) in *)
    (* let bstr = Bytes.to_string (Bytes.create(5)) in *)
    let bstr = Bytes.create len in
    really_input file bstr 0 len;
    vocab := (Bytes.to_string bstr) :: !vocab;
  done;

  (List.rev !vocab, List.rev !vocab_scores, !max_token_length)
;;


type run_state = {
  mutable x : float array;
  mutable xb : float array;
  mutable q : float array;
  mutable k : float array;
  mutable v : float array;
  mutable att : float array;
  mutable key_cache : float array;
  mutable value_cache : float array;
  mutable xb2 : float array;
  mutable hb : float array;
  mutable hb2 : float array;
  mutable logits : float array;
}

(* let init_run_state state config =
  state.x <- Array.init config.dim (fun _ -> 0.);
  state.xb <- Array.init config.dim (fun _ -> 0.);
  state.xb2 <- Array.init config.dim (fun _ -> 0.);
  state.hb <- Array.init config.hidden_dim (fun _ -> 0.);
  state.hb2 <- Array.init config.hidden_dim (fun _ -> 0.);
  state.q <- Array.init config.dim (fun _ -> 0.);
  state.k <- Array.init config.dim (fun _ -> 0.);
  state.v <- Array.init config.dim (fun _ -> 0.);
  state.att <- Array.init (config.n_heads * config.seq_len) (fun _ -> 0.);
  state.logits <- Array.init config.vocab_size (fun _ -> 0.);
  state.key_cache <- Array.init (config.n_layers * config.seq_len * config.dim) (fun _ -> 0.);
  state.value_cache <- Array.init (config.n_layers * config.seq_len * config.dim) (fun _ -> 0.)
;; *)
  
let init_run_state state config =
  state.x <- Array.make config.dim 0.;
  state.xb <- Array.make config.dim 0.;
  state.xb2 <- Array.make config.dim 0.;
  state.hb <- Array.make config.hidden_dim 0.;
  state.hb2 <- Array.make config.hidden_dim 0.;
  state.q <- Array.make config.dim 0.;
  state.k <- Array.make config.dim 0.;
  state.v <- Array.make config.dim 0.;
  state.att <- Array.make (config.n_heads * config.seq_len) 0.;
  state.logits <- Array.make config.vocab_size 0.;
  state.key_cache <- Array.make (config.n_layers * config.seq_len * config.dim) 0.;
  state.value_cache <- Array.make (config.n_layers * config.seq_len * config.dim) 0.
;;


let rec print_int_list = function
  | [] -> print_endline "";
  | x::xs -> print_int x;
    if xs <> [] then print_string ", ";
    print_int_list xs


let print_float_array arr =
  let len = Array.length arr in
  for i = 0 to len - 1 do
    print_float arr.(i);
    if i < len - 1 then
      print_string ", ";
  done;
  print_endline ""
;;


let rec take n lst =
  match n, lst with
  | 0, _ | _, [] -> []
  | n, x :: xs -> x :: take (n - 1) xs

let rec drop n lst =
  match n, lst with
  | 0, _ | _, [] -> lst
  | n, _ :: xs -> drop (n - 1) xs

let bpe_encode text vocab vocab_scores =
  let tokens = ref [] in

  let rec index_opt elem lst =
    let rec loop idx = function
      | [] -> None
      | x :: xs -> if x = elem then Some idx else loop (idx + 1) xs
    in
    loop 0 lst
  in

  (* Helper function to look up a string in the vocab *)
  let str_lookup str vocab =
    match index_opt str vocab with
    | Some id -> id
    | None -> -1
  in

  (* Encode individual characters in the input text *)
  String.iteri (fun i -> fun char ->
    let string = String.make 1 char in
    let id = str_lookup string vocab in
    if id = -1 then begin
      Printf.printf "not a good prompt at pos %d\n" i;
      exit 1
    end;
    tokens := id :: !tokens
  ) text;

  tokens := List.rev !tokens;

  (* print_int_list !tokens; *)

  let vocab_a = Array.of_list vocab in
  let vocab_scores_a = Array.of_list vocab_scores in

  let rec merge_tokens tokens vocab vocab_scores =
    let rec find_best_pair tokens best_score best_id best_index i = match tokens with
      | token1::token2::ts -> 
        let string = vocab_a.(token1) ^ vocab_a.(token2) in
        let id = str_lookup string vocab in
        let best_score, best_id, best_index =
          if id = -1 then best_score, best_id, best_index
          else let score = vocab_scores_a.(id) in 
            if score < best_score then best_score, best_id, best_index
            else score, id, i
        in
        find_best_pair (token2::ts) best_score best_id best_index (i+1)

      | token::[] -> best_id, best_index
      | [] -> best_id, best_index
    in
      let best_id, best_index = (find_best_pair tokens (-1e10) (-1) 0 0) in
      if best_id = -1
        then tokens
        else merge_tokens ((take best_index tokens) @ (best_id :: drop (best_index + 2) tokens)) vocab vocab_scores

  in
  let t2 = merge_tokens !tokens vocab vocab_scores in
  print_int_list t2;
  !tokens
;;


let rmsnorm out x weight =
  let size = Array.length x in
  let ss = Array.fold_left (fun acc elem -> acc +. elem *. elem) 0.0 x /. float size +. 1e-5 in
  let ss = 1.0 /. sqrt ss in
  Array.mapi (fun i elem -> weight.(i) *. ss *. elem) x
;;

let matmul xout x w n d =
  Array.init d (fun i ->
    let val' =
      Array.init n (fun j -> w.(i * n + j) *. x.(j))
      |> Array.fold_left (+.) 0.0
    in
    val'
  )
;;

let dot_product arr1 arr2 =
  let multiplied = Array.map2 ( *. ) arr1 arr2 in
  Array.fold_left ( +. ) 0.0 multiplied
;;

let softmax x size =
  let max_val = Array.fold_left max x.(0) (Array.sub x 1 (size - 1)) in
  let exp_sum, softmaxed, index =
    Array.fold_left (fun (sum, result, index) elem ->
      if index < size then
        let exp_elem = exp (elem -. max_val) in
        sum +. exp_elem,
        Array.append result [|exp_elem|],
        index + 1
      else
        sum,
        Array.append result [|elem|],
        index + 1
    ) (0.0, [||], 0) x
  in
  let softmaxed =
    Array.mapi (fun i elem ->
      if i < size then elem /. exp_sum else elem
    ) softmaxed
  in
  softmaxed
;;

let accum a b =
  Array.iteri (fun i elem -> a.(i) <- a.(i) +. elem) b;
  a



let transformer token pos conf state weights =
  let dim = conf.dim in
  let hidden_dim = conf.hidden_dim in
  let head_size = dim / conf.n_heads in
  print_endline "transformer";

  (* Copy the token embedding into x *)
  let content_row = Array.sub weights.token_embedding_table (token * dim) dim in
  Array.blit content_row 0 state.x 0 dim;

  (* print_float content_row.((token * dim) - 1); *)
  (* print_float 0.; *)

  (* Pluck out the "pos" row of freq_cis_real and freq_cis_imag *)
  let freq_cis_real_row =
    Array.sub weights.freq_cis_real
              (pos * (head_size / 2))
              (head_size / 2) in

  let freq_cis_imag_row =
    Array.sub weights.freq_cis_imag
              (pos * (head_size / 2))
              (head_size / 2) in

  (* print_float freq_cis_real_row.(10); *)
  print_float_array freq_cis_real_row;
  print_float_array freq_cis_imag_row;
  (* print_float_array content_row; *)

  (* Forward all the layers *)
  for l = 0 to (conf.n_layers - 1) do
    (* for l = 0 to 0 do *)
    (* Attention rmsnorm *)
    state.xb <- rmsnorm state.xb state.x (Array.sub weights.rms_att_weight (l * dim) (dim));

    (* QKV matmuls for this position *)
    state.q <- matmul state.q state.xb
                      (Array.sub weights.wq (l * dim * dim) (dim * dim)) dim dim;

    state.k <- matmul state.k state.xb
                      (Array.sub weights.wk (l * dim * dim) (dim * dim)) dim dim;

    state.v <- matmul state.v state.xb
                      (Array.sub weights.wv (l * dim * dim) (dim * dim)) dim dim;

    (* Apply RoPE rotation to the q and k vectors for each head *)
    for h = 0 to (conf.n_heads - 1) do
      (* Get the q and k vectors for this head *)
      let q = Array.sub state.q (h * head_size) head_size in
      let k = Array.sub state.k (h * head_size) head_size in

      let rotate_qk freq_cis_real_row freq_cis_imag_row q k head_size =
        let rec rotate i =
          if i < head_size then
            let q0 = q.(i) in
            let q1 = q.(i + 1) in
            let k0 = k.(i) in
            let k1 = k.(i + 1) in
            let fcr = freq_cis_real_row.(i / 2) in
            let fci = freq_cis_imag_row.(i / 2) in
            q.(i) <- q0 *. fcr -. q1 *. fci;
            q.(i + 1) <- q0 *. fci +. q1 *. fcr;
            k.(i) <- k0 *. fcr -. k1 *. fci;
            k.(i + 1) <- k0 *. fci +. k1 *. fcr;
            rotate (i + 2)
        in
        rotate 0 in
      rotate_qk freq_cis_real_row freq_cis_imag_row q k head_size;

      (* Reassigned back to state.q and state.k *)
      for i = 0 to (head_size - 1) do
        state.q.(h * head_size + i) <- q.(i);
        state.k.(h * head_size + i) <- k.(i);
      done;
    done;

    (* Save key and value at this time step (pos) to our kv cache *)
    let loff = l * conf.seq_len * dim in
    let offset = loff + pos * dim in
    Array.blit state.k 0 state.key_cache offset dim;
    Array.blit state.v 0 state.value_cache offset dim;

    (* Multihead attention. Iterate over all heads *)
    for h = 0 to (conf.n_heads - 1) do
      (* Get the query vector for this head *)
      let q = Array.sub state.q (h * head_size) head_size in

      (* Attention scores for this head *)
      let att = Array.sub state.att (h * conf.seq_len) conf.seq_len in

      (* Iterate over all timesteps, including the current one *)
      for t = 0 to pos do
        (* Get the key vector for this head and at this timestep *)
        let k =
          Array.sub state.key_cache (loff + t * dim + h * head_size) head_size
        in

        (* Calculate the attention score as the dot product of q and k *)
        let score = dot_product q k /. sqrt (float_of_int head_size) in

        (* Save the score to the attention buffer *)
        att.(t) <- score;
      done;

      let att = softmax att (pos + 1) in

      let xb_ptr = h * head_size in
      let zero_head_size = Array.make head_size 0. in
      Array.blit zero_head_size 0 state.xb xb_ptr head_size;

      for t = pos downto 0 do
        (* Get the value vector for this head and at this timestep *)
        let v = Array.sub state.value_cache (loff + t * dim + h * head_size) head_size in
        (* Get the attention weight for this timestep *)
        let a = att.(t) in
        (* Accumulate the weighted value into xb *)
        for i = 0 to (head_size - 1) do
          state.xb.(xb_ptr + i) <- state.xb.(xb_ptr + i) +. a *. v.(i);
        done;
      done;
    done;

    state.xb2 <- matmul state.xb2 state.xb
                  (Array.sub weights.wo (l * dim * dim) (dim * dim)) dim dim;

    state.x <- accum state.x state.xb2;

    state.xb <- rmsnorm state.xb state.x
                     (Array.sub weights.rms_ffn_weight (l * dim) dim);

    state.hb <- matmul state.hb state.xb
                     (Array.sub weights.w1 (l * dim * hidden_dim) (dim * hidden_dim)) dim hidden_dim;


    state.hb2 <- matmul state.hb2 state.xb
                      (Array.sub weights.w3 (l * dim * hidden_dim) (dim * hidden_dim)) dim hidden_dim;

    state.hb <- Array.init hidden_dim (fun i ->
      state.hb.(i) *. (1.0 /. (1.0 +. exp (-. state.hb.(i))))
    );

    state.hb <- Array.init hidden_dim (fun i -> state.hb.(i) *. state.hb2.(i));

    state.xb <- matmul state.xb state.hb
                    (Array.sub weights.w2 (l * dim * hidden_dim) (dim * hidden_dim)) hidden_dim dim;

    state.x <- accum state.x state.xb; 
    (* print_float_array state.x *)
  done;

  state.x <- rmsnorm state.x state.x weights.rms_final_weight;

  state.logits <- matmul state.logits state.x weights.wcls dim conf.vocab_size;

  (* print_float_array state.logits *)
  


;;




let run args =
  let checkpoint = args.checkpoint in
  let temperature = args.temperature in
  let steps = args.steps in
  let prompt = args.prompt in


  let rng_seed = int_of_float (Unix.time ()) in
  Random.init rng_seed;
  print_endline (string_of_int rng_seed);
  let file = open_in_bin checkpoint in
  let config = read_config file checkpoint in
  print_config config;
  let stat = Unix.stat checkpoint in
  let file_size = stat.st_size in
  (* print_endline (string_of_int file_size) *)
  let weights = create_transformer_weights () in 
  checkpoint_init_weights weights config file config.shared_weights file_size;
  (* print_endline transformer_weights.token_embedding_table *)
  (* print_token_embedding_table transformer_weights *)
  (* print_endline (string_of_float transformer_weights.token_embedding_table.(0));
  print_endline (string_of_float transformer_weights.token_embedding_table.(1));
  print_endline (string_of_float transformer_weights.token_embedding_table.(2));
  print_endline (string_of_float transformer_weights.token_embedding_table.(10000));
  print_endline (string_of_float transformer_weights.token_embedding_table.(100000));
  print_endline (string_of_float transformer_weights.freq_cis_imag.(1000)) *)

  let steps = if steps <= 0 || steps > config.seq_len then config.seq_len else steps in
  (* print_endline (string_of_int steps); *)

  let tokenizer_file = open_in_bin "tokenizer.bin" in
  let vocab, vocab_scores, max_token_length = tokenizer_init config tokenizer_file in
  (* print_endline "filhuksdf";
  print_endline (string_of_int max_token_length);
  print_endline (string_of_float (List.nth vocab_scores 0));
  print_endline (string_of_float (List.nth vocab_scores 1000));
  print_endline (string_of_float (List.nth vocab_scores 1120));
  print_endline (string_of_float (List.nth vocab_scores 1137));
  print_endline (Bytes.to_string (List.nth vocab 0));
  print_endline (Bytes.to_string (List.nth vocab 1));
  print_endline (Bytes.to_string (List.nth vocab 2));
  print_endline (Bytes.to_string (List.nth vocab 3));
  print_bytes (List.nth vocab 0);
  print_bytes (List.nth vocab 1);
  print_bytes (List.nth vocab 2);
  print_bytes (List.nth vocab 3); *)

  (* print_endline (List.nth vocab 0); *)




  let state = {
    x = [||];
    xb = [||];
    q = [||];
    k = [||];
    v = [||];
    att = [||];
    key_cache = [||];
    value_cache = [||];
    xb2 = [||];
    hb = [||];
    hb2 = [||];
    logits = [||];
  } in
  init_run_state state config;


  let prompt_tokens =
    if String.length prompt > 0 then
      bpe_encode prompt vocab vocab_scores
    else
      []
    in prompt_tokens;

  (* Start the main loop *)
  let start = ref 0 in  (* Used to time our code, only initialized after the first iteration *)
  let next_token = ref 0 in (* Will store the next token in the sequence *)
  (* Initialize with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer *)
  let token = ref 1 in
  let pos = ref 0 in (* Position in the sequence *)
  (* Explicitly print the initial BOS token for stylistic symmetry reasons *)
  print_endline "<s>";

  (* transformer token pos config state weights; *)

  transformer !token !pos config state weights;
  print_int !pos;

  if !pos < List.length prompt_tokens then
    next_token := List.nth prompt_tokens !pos
  else
    let () =
      print_endline (string_of_float state.logits.(0))
      (* print_endline "q" *)
      (* if temperature = 0.0 then
        next_token := argmax state.logits
      else
        let () =
          let factor = 1.0 /. temperature in
          for i = 0 to Array.length state.logits - 1 do
            state.logits.(i) <- state.logits.(i) *. factor;
          done;
        in
        softmax state.logits config.vocab_size;
        next_token := sample state.logits *)
    in
    ();



  if temperature = -1. && steps == -2821 && prompt = "just to stop [unused-var] warnings" then print_endline "skfjkhg"
  

  (* print_endline checkpoint;
  let file_name = "tokenizer.bin" in
  try
    let file = open_in_bin file_name in
    print_endline (string_of_int (input_binary_int file));
    print_endline (string_of_int (input_binary_int file));
    print_endline (string_of_int (input_binary_int file));
    print_endline (string_of_bool (not (input_binary_int file = 0)));
    if not (input_binary_int file = 0) then begin
      close_in file;
      print_endline "Couldn't load tokenizer.bin";
      exit 1
    end;

    let vocab, vocab_scores, max_token_length = tokenizer_init config file in

    (* Use vocab, vocab_scores, and max_token_length as needed *)

    close_in file;
  with
  | Sys_error msg ->
    print_endline ("Error: " ^ msg);
    exit 1
;; *)




let () =
  let args =
    { checkpoint = "";
      temperature = 0.0;
      steps = 256;
      prompt = "" }
  in
  if Array.length Sys.argv < 2 then begin
    print_endline "Usage: ocaml llama2.ml <checkpoint_file> [temperature] [steps] [prompt]";
    exit 1
  end;

  args.checkpoint <- Sys.argv.(1);

  if Array.length Sys.argv >= 3 then
    args.temperature <- float_of_string Sys.argv.(2);

  if Array.length Sys.argv >= 4 then
    args.steps <- int_of_string Sys.argv.(3);

  if Array.length Sys.argv >= 5 then
    args.prompt <- Sys.argv.(4);

  run args
