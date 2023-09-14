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
}

(* let checkpoint_init_weights weights conf file shared_weights file_size =
  (* let ic = open_in_bin checkpoint in *)
  let read_floats count =
    let buffer = Bytes.create (count * 4) in
    let _ = input file buffer 0 (count * 4) in
    let values = Array.init count (fun i ->
      let offset = i * 4 in
      let bytes = Bytes.sub buffer offset 4 in
      let float_bits = 
        Int32.logor (Int32.of_int (Char.code (Bytes.get bytes 0)))
                    (Int32.shift_left (Int32.of_int (Char.code (Bytes.get bytes 1))) 8)
      in
      Int32.float_of_bits float_bits
    ) in
    values
  in

  weights.token_embedding_table <- read_floats (conf.vocab_size * conf.dim);
  weights.rms_att_weight <- read_floats (conf.n_layers * conf.dim)
;; *)


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

  
(* let int_size_32 = 32 - 1
let int2bin_32 =
  let buf = Bytes.create int_size_32 in
  fun n ->
    for i = 0 to int_size_32 - 1 do
      let pos = int_size_32 - 1 - i in
      Bytes.set buf pos (if n land (1 lsl i) != 0 then '1' else '0')
    done;
    (* skip leading zeros *)
    (* match Bytes.index_opt buf '1' with
    | None -> "0b0"
    | Some i -> "0b" ^ Bytes.sub_string buf i (int_size_32 - i) *)
    Bytes.sub_string buf 0 int_size_32 *)

let int32_size = 32

let int32_to_bin n =
  let buf = Bytes.create int32_size in
  for i = 0 to int32_size - 1 do
    let pos = int32_size - 1 - i in
    Bytes.set buf pos (if Int32.logand n (Int32.shift_left (Int32.of_int 1) i) <> 0l then '1' else '0')
  done;
  Bytes.sub_string buf 0 int32_size

let input_binary_float file =
  let int_bits = input_binary_int file in
  print_endline (int2bin int_bits);
  (* let q = 0b000000000000000000000000000000011111111111111111111111111111111 land int_bits in *)
  let q = int_bits in
  let q32 = Int32.of_int q in
  print_endline (int32_to_bin q32);
  print_endline (int2bin q);
  (* print_endline (string_of_float (Int32.float_of_bits (Int32.of_int q))); *)
  print_endline (string_of_float (Int32.float_of_bits q32));
  let float_bits = Int32.float_of_bits (Int32.of_int int_bits) in
  float_bits
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
  weights.rms_att_weight <- read_floats (conf.n_layers * conf.dim)



let create_transformer_weights () =
  {
    token_embedding_table = [||];
    rms_att_weight = [||];
  }


let print_token_embedding_table weights =
  Array.iteri (fun i value ->
    Printf.printf "token_embedding_table[%d]: %f\n" i value
  ) weights.token_embedding_table
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
  let transformer_weights = create_transformer_weights () in 
  checkpoint_init_weights transformer_weights config file file_size config.shared_weights;
  (* print_endline transformer_weights.token_embedding_table *)
  (* print_token_embedding_table transformer_weights *)
  print_endline (string_of_float transformer_weights.token_embedding_table.(0));
  print_endline (string_of_float transformer_weights.token_embedding_table.(1));
  print_endline (string_of_float transformer_weights.token_embedding_table.(2));
  print_endline (string_of_float transformer_weights.token_embedding_table.(10000))



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
