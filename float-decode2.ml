(* 

let q = Int32.of_int 0b10111101011101010000101010110000 in
print_endline (string_of_float (Int32.float_of_bits q)) *)



(* let w =  (Int32.of_int 0b10111101011101010000101010110000) in
let e = Int32.logand (Int32.of_int 0b111111110000000000000000) w in
print_endline (string_of_float (Int32.float_of_bits e)) *)



let int32_size = 32

let int32_to_bin n =
  let buf = Bytes.create int32_size in
  for i = 0 to int32_size - 1 do
    let pos = int32_size - 1 - i in
    Bytes.set buf pos (if Int32.logand n (Int32.shift_left (Int32.of_int 1) i) <> 0l then '1' else '0')
  done;
  Bytes.sub_string buf 0 int32_size
;;
(* 
let w =  (Int32.of_int 0b10111101011101010000101010110000) in
(* let e = Int32.logand (Int32.of_int 0b111111110000000000000000) w in *)
let e = Int32.logand (Int32.of_int 0b11111111111111111111111111111111) w in
print_endline (string_of_float (Int32.float_of_bits e));
print_endline (string_of_float (Int32.float_of_bits w)) *)



(* let w =  (Int32.of_int 0b10110000000010100111010110111101) in
let p1 = Int32.shift_left (Int32.logand (Int32.shift_left(Int32.of_int 0b11111111) 0) w) 24 in
let p2 = Int32.shift_left (Int32.logand (Int32.shift_left(Int32.of_int 0b11111111) 8) w) 8 in
let p3 = Int32.shift_left (Int32.logand (Int32.shift_right(Int32.of_int 0b11111111) 16) w) 8 in
let p4 = Int32.shift_left (Int32.logand (Int32.shift_right(Int32.of_int 0b11111111) 24) w) 24 in
let e = Int32.add (Int32.add p1 p2) (Int32.add p3 p4) in
print_endline (string_of_float (Int32.float_of_bits e));
print_endline (int32_to_bin e) *)


let float_decode bits =
  let p1 = Int32.shift_left (Int32.logand (Int32.shift_left(Int32.of_int 0b11111111) 0) bits) 24 in
  let p2 = Int32.shift_left (Int32.logand (Int32.shift_left(Int32.of_int 0b11111111) 8) bits) 8 in
  let p3 = Int32.shift_right (Int32.logand (Int32.shift_left(Int32.of_int 0b11111111) 16) bits) 8 in
  let p4 = Int32.shift_right (Int32.logand (Int32.shift_left(Int32.of_int 0b11111111) 24) bits) 24 in
  let sum = Int32.add (Int32.add p1 p2) (Int32.add p3 p4) in
  Int32.float_of_bits sum
;;


print_endline (string_of_float (float_decode (Int32.of_int 0b10110000000010100111010110111101)));