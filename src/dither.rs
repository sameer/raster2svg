// https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
// let dither = {
//     let mut dither = image.clone();
//     for k in 0..=3 {
//         for j in 0..dither.shape()[2] {
//             for i in 0..dither.shape()[1] {
//                 let original_value = dither[[k, i, j]];
//                 let new_value = if original_value >= 128 {
//                     u8::MAX
//                 } else {
//                     u8::MIN
//                 };
//                 dither[[k, i, j]] = new_value;
//                 const OFFSETS: [[isize; 2]; 4] = [[1, 0], [-1, 1], [0, 1], [1, 1]];
//                 const QUANTIZATION: [u16; 4] = [7, 3, 5, 1];
//                 let (errs, add) = if original_value > new_value {
//                     let mut quantization_errors = [0; 4];
//                     for (idx, q) in QUANTIZATION.iter().enumerate() {
//                         quantization_errors[idx] =
//                             ((q * ((original_value - new_value) as u16)) / 16) as u8;
//                     }
//                     (quantization_errors, true)
//                 } else {
//                     let mut quantization_errors = [0; 4];
//                     for (idx, q) in QUANTIZATION.iter().enumerate() {
//                         quantization_errors[idx] =
//                             ((q * ((new_value - original_value) as u16)) / 16) as u8;
//                     }
//                     (quantization_errors, false)
//                 };
//                 for (offset, err) in OFFSETS.iter().zip(errs.iter()) {
//                     let index = [
//                         k as isize,
//                         (i as isize + offset[0]).clamp(-1, (dither.shape()[1] - 1) as isize),
//                         (j as isize + offset[1]).clamp(-1, (dither.shape()[2] - 1) as isize),
//                     ];
//                     let value = if add {
//                         dither
//                             .slice(s![index[0], index[1], index[2]])
//                             .into_scalar()
//                             .saturating_add(*err)
//                     } else {
//                         dither
//                             .slice(s![index[0], index[1], index[2]])
//                             .into_scalar()
//                             .saturating_sub(*err)
//                     };
//                     dither
//                         .slice_mut(s![index[0], index[1], index[2]])
//                         .fill(value);
//                 }
//             }
//         }
//     }
//     dither
// };
