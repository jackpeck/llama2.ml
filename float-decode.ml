let rec split_string str chunk_size =
  let rec aux acc str =
    if String.length str = 0 then
      List.rev acc
    else
      let chunk, rest = String.split_at chunk_size str in
      aux (chunk :: acc) rest
  in
  aux [] str

let decode_float bit_string =
  if String.length bit_string mod 32 <> 0 then
    failwith "Bit string length is not a multiple of 32";
  
  let chunks = split_string bit_string 32 in
  let result = List.map (fun chunk ->
    let bytes = List.map (fun i -> int_of_string ("0b" ^ i)) (split_string chunk 8) in
    let float_bytes = Bytes.of_list bytes in
    let float_value = Bytes.to_float float_bytes in
    float_value
  ) chunks in
  result

let bit_string = "10110000000010100111010110111101";;
let decoded_floats = decode_float bit_string;;

List.iter (Printf.printf "%f ") decoded_floats;;
