use cairo::Context;
use image::io::Reader as ImageReader;
use log::*;
use ndarray::prelude::*;
use rand::{distributions::Uniform, prelude::*};
use spade::delaunay::IntDelaunayTriangulation;
use std::{
    env,
    io::{self, Read},
    path::PathBuf,
};
use structopt::StructOpt;
use uom::si::f64::Length;
use uom::si::length::{inch, millimeter};

mod mst;
mod tsp;
mod voronoi;

#[derive(Debug, StructOpt)]
#[structopt(author, about)]
struct Opt {
    /// A path to an image, else reads from stdin
    file: Option<PathBuf>,

    /// Determines the scaling of the output SVG
    #[structopt(long, default_value = "96")]
    dots_per_inch: f64,

    /// Diameter of the pen stroke in units of your choice
    #[structopt(long, default_value = "1 mm")]
    pen_diameter: Length,

    /// Output file path (overwrites old files), else writes to stdout
    #[structopt(short, long)]
    out: Option<PathBuf>,
}

const RED: [f64; 3] = [1., 0., 0.];
const GREEN: [f64; 3] = [0., 1., 0.];
const BLUE: [f64; 3] = [0., 0., 1.];

fn main() -> io::Result<()> {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "raster2svg=info")
    }
    env_logger::init();
    let opt = Opt::from_args();

    let (pen_image, pen_point_counts) = {
        let image = match opt.file {
            Some(filepath) => ImageReader::open(filepath)?.decode(),
            None => {
                info!("Reading from stdin");
                let mut bytes = vec![];
                io::stdin().read_to_end(&mut bytes)?;
                let cursor = io::Cursor::new(bytes);
                ImageReader::new(cursor).decode()
            }
        }
        .expect("not an image")
        .to_rgb16();

        let image = Array::from_iter(
            image
                .pixels()
                .map(|p| p.0.iter().copied())
                .flatten()
                .map(|p| p as f64 / u16::MAX as f64),
        )
        .into_shape((image.height() as usize, image.width() as usize, 3))
        .unwrap()
        .reversed_axes();

        let mut pen_image = Array::zeros((0, image.shape()[1], image.shape()[2]));

        [RED, GREEN, BLUE].iter().for_each(|color| {
            pen_image
                .push(
                    Axis(0),
                    image
                        .map_axis(Axis(0), |pixel| {
                            let val = (1.
                                - pixel
                                    .iter()
                                    .zip(color)
                                    .map(|(x1, x2)| (x1 - x2).powi(2))
                                    .sum::<f64>()
                                    .sqrt()
                                    / 3.0_f64.sqrt())
                            .powi(2);
                            if val < 0.2 {
                                0.0
                            } else {
                                val
                            }
                        })
                        .view(),
                )
                .unwrap();
        });

        // Add inverse lightness from D65 sRGB to D50 CIE XYZ to D50 CIE LAB
        pen_image
            .push(
                Axis(0),
                image
                    .map_axis(Axis(0), |pixel| {
                        const LUMINANCE_COEFFICIENTS: [f64; 3] = [0.2225045, 0.7168786, 0.0606169];
                        let y = pixel
                            .iter()
                            .copied()
                            .map(|component| {
                                if component <= 0.04045 {
                                    component / 12.92
                                } else {
                                    ((component + 0.055) / 1.055).powf(2.4)
                                }
                            })
                            .zip(&LUMINANCE_COEFFICIENTS)
                            .map(|(component, coefficient)| component * coefficient)
                            .sum::<f64>()
                            .clamp(0., LUMINANCE_COEFFICIENTS.iter().sum());

                        const DELTA: f64 = 6. / 29.;
                        let l =
                            116. * if y > DELTA.powi(3) {
                                y.cbrt()
                            } else {
                                y / (3. * DELTA.powi(2)) + 4. / 29.
                            } - 16.;

                        // 1. - (pixel
                        //     .iter()
                        //     .max_by(|a, b| a.partial_cmp(&b).unwrap())
                        //     .unwrap())
                        // 1. - (pixel
                        //     .iter()
                        //     .max_by(|a, b| a.partial_cmp(&b).unwrap())
                        //     .unwrap()
                        //     + pixel
                        //         .iter()
                        //         .min_by(|a, b| a.partial_cmp(&b).unwrap())
                        //         .unwrap())
                        //     / 2.

                        // let dist = pixel.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
                        //     / 3.0_f64.sqrt();
                        // if dist
                        //     < [RED, GREEN, BLUE]
                        //         .iter()
                        //         .map(|color| {
                        //             pixel
                        //                 .iter()
                        //                 .zip(color)
                        //                 .map(|(x1, x2)| (x1 - x2).powi(2))
                        //                 .sum::<f64>()
                        //                 .sqrt()
                        //                 / 3.0_f64.sqrt()
                        //         })
                        //         .max_by(|a, b| a.partial_cmp(b).unwrap())
                        //         .unwrap()
                        // {
                        (1. - (l.clamp(0., 100.) / 100.)).powi(2)
                        // } else {
                        //     0.
                        // }

                        // 1. - pixel.iter().sum::<f64>() / 3.
                    })
                    .view(),
            )
            .unwrap();

        // https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
        let dither = {
            let mut dither = image.clone();
            for k in 0..=2 {
                for j in 0..dither.shape()[2] {
                    for i in 0..dither.shape()[1] {
                        let original_value = dither[[k, i, j]];
                        let new_value = if original_value >= 0.5 { 1.0 } else { 0.0 };
                        dither[[k, i, j]] = new_value;
                        const OFFSETS: [[isize; 2]; 4] = [[1, 0], [-1, 1], [0, 1], [1, 1]];
                        const QUANTIZATION: [f64; 4] = [7., 3., 5., 1.];
                        let (errs, add) = if original_value > new_value {
                            let mut quantization_errors = [0.0; 4];
                            for (idx, q) in QUANTIZATION.iter().enumerate() {
                                quantization_errors[idx] = q * (original_value - new_value) / 16.;
                            }
                            (quantization_errors, true)
                        } else {
                            let mut quantization_errors = [0.0; 4];
                            for (idx, q) in QUANTIZATION.iter().enumerate() {
                                quantization_errors[idx] = q * (original_value - new_value) / 16.;
                            }
                            (quantization_errors, false)
                        };
                        for (offset, err) in OFFSETS.iter().zip(errs.iter()) {
                            let index = [
                                k as isize,
                                (i as isize + offset[0])
                                    .clamp(-1, (dither.shape()[1] - 1) as isize),
                                (j as isize + offset[1])
                                    .clamp(-1, (dither.shape()[2] - 1) as isize),
                            ];
                            let value = if add {
                                dither.slice(s![index[0], index[1], index[2]]).into_scalar() - *err
                            } else {
                                dither.slice(s![index[0], index[1], index[2]]).into_scalar() - *err
                            };
                            dither
                                .slice_mut(s![index[0], index[1], index[2]])
                                .fill(value);
                        }
                    }
                }
            }
            dither
        };

        let mut pen_point_counts = [0, 0, 0, 0];
        for pixel in dither.lanes(Axis(0)) {
            for (i, value) in pixel.iter().enumerate() {
                if *value == 1.0 {
                    pen_point_counts[i] += 1;
                }
            }
        }
        pen_point_counts[3] = pen_point_counts[0..=2].iter().sum::<usize>() / 3;
        dbg!(pen_point_counts);
        pen_point_counts.iter_mut().for_each(|x| *x /= 8);
        // dbg!(&pen_point_counts);
        (pen_image, pen_point_counts)
    };

    let pen_diameter = opt.pen_diameter.get::<millimeter>();
    let in_to_mm = Length::new::<inch>(1.).get::<millimeter>();
    let width = (pen_image.shape()[1] as f64 / opt.dots_per_inch * in_to_mm).round();
    let height = (pen_image.shape()[2] as f64 / opt.dots_per_inch * in_to_mm).round();

    let mut surf = match &opt.out {
        Some(_) => cairo::SvgSurface::new(width, height, opt.out).unwrap(),
        None => cairo::SvgSurface::for_stream(width, height, std::io::stdout()).unwrap(),
    };
    surf.set_document_unit(cairo::SvgUnit::Mm);
    let ctx = Context::new(&surf);

    ctx.set_source_rgb(1., 1., 1.);
    ctx.rectangle(0., 0., width, height);
    ctx.fill();

    ctx.set_line_width(pen_diameter);
    // Makes life easier to work with the same coords as the image
    ctx.scale(
        width / pen_image.shape()[1] as f64,
        height / pen_image.shape()[2] as f64,
    );

    for k in [3usize, 0, 1, 2].iter().copied() {
        info!("Processing {}", k);
        let mut voronoi_points = {
            // let mut indices = dither
            //     .slice(s![k, .., ..])
            //     .indexed_iter()
            //     .filter_map(|idx_and_elem: ((usize, usize), &u8)| {
            //         if *idx_and_elem.1 == u8::MAX {
            //             Some([idx_and_elem.0 .0 as i64, idx_and_elem.0 .1 as i64])
            //         } else {
            //             None
            //         }
            //     })
            //     .collect::<Vec<_>>();
            // indices.shuffle(&mut rng);
            // indices.truncate(indices.len().min(100000));
            // indices

            let mut rng = thread_rng();
            // (0..20000)
            //     .map(|_| {
            //         [
            //             rng.gen_range(0..image.shape()[1]),
            //             rng.gen_range(0..image.shape()[2]),
            //         ]
            //     })
            //     .collect::<Vec<_>>()
            let mut points = Vec::with_capacity(pen_point_counts[k]);
            let distribution = Uniform::from(0.0..1.);
            while points.len() < points.capacity() {
                let x = rng.gen_range(0..pen_image.shape()[1]);
                let y = rng.gen_range(0..pen_image.shape()[2]);
                let pick_probability = distribution.sample(&mut rng);
                if pen_image[[k, x, y]] > pick_probability {
                    points.push([x, y]);
                }
            }
            points
        };
        if voronoi_points.len() < 3 {
            warn!(
                "Color channel {} has too few vertices ({}) to draw, skipping",
                k,
                voronoi_points.len()
            );
            continue;
        }

        let mut colored_pixels;
        let mut last_average_distance: Option<f64> = None;
        loop {
            debug!("Jump flooding voronoi");
            colored_pixels = voronoi::compute_voronoi(
                &voronoi_points,
                pen_image.shape()[1],
                pen_image.shape()[2],
            );

            debug!("Secord's algorithm");
            let expected_assignment_capacity =
                pen_image.shape()[1] * pen_image.shape()[2] / voronoi_points.len();
            let mut point_assignments =
                vec![
                    Vec::<[usize; 2]>::with_capacity(expected_assignment_capacity);
                    voronoi_points.len()
                ];
            for i in 0..pen_image.shape()[1] as usize {
                for j in 0..pen_image.shape()[2] as usize {
                    point_assignments[colored_pixels[j][i]].push([i, j]);
                }
            }

            let centroids: Vec<[usize; 2]> = point_assignments
                .iter()
                .map(|point_assignment| {
                    let denominator = point_assignment.iter().fold(0., |mut acc, [x, y]| {
                        acc += pen_image[[k, *x, *y]];
                        acc
                    });
                    let numerator_y = point_assignment.iter().fold(0., |mut acc, [x, y]| {
                        acc += *y as f64 * pen_image[[k, *x, *y]];
                        acc
                    });
                    let numerator_x = point_assignment.iter().fold(0., |mut acc, [x, y]| {
                        acc += *x as f64 * pen_image[[k, *x, *y]];
                        acc
                    });
                    [numerator_x / denominator, numerator_y / denominator]
                })
                .map(|[x, y]| {
                    [
                        (x.round() as usize).clamp(0, pen_image.shape()[1] - 1),
                        (y.round() as usize).clamp(0, pen_image.shape()[2] - 1),
                    ]
                })
                .collect();

            debug!("Check stopping condition");
            // stopping condition: average distance moved by all stippling points is small
            let distance_sum = voronoi_points
                .iter()
                .zip(centroids.iter())
                .map(|(voronoi, centroid)| abs_distance_squared(*voronoi, *centroid))
                .map(|distance_squared| (distance_squared as f64))
                .sum::<f64>()
                .sqrt();
            voronoi_points = centroids;
            let average_distance = distance_sum / voronoi_points.len() as f64;
            debug!("at {}", average_distance);
            if let Some(last_average_distance) = last_average_distance {
                if (last_average_distance - average_distance).abs() <= f64::EPSILON {
                    break;
                }
            }
            last_average_distance = Some(average_distance)
        }

        // On the off chance 2 points end up being the same...
        voronoi_points.sort_unstable();
        voronoi_points.dedup();

        let mut delaunay = IntDelaunayTriangulation::with_tree_locate();
        for vertex in &voronoi_points {
            delaunay.insert([vertex[0] as i64, vertex[1] as i64]);
        }

        let tree = crate::mst::compute_mst(&voronoi_points, &delaunay);
        let tsp = crate::tsp::approximate_tsp_with_mst(&voronoi_points, &tree);

        debug!("Draw to svg");

        if k == 0 || k == 5 {
            ctx.set_source_rgb(1., 0., 0.);
        } else if k == 1 || k == 6 {
            ctx.set_source_rgb(0., 1., 0.);
        } else if k == 2 || k == 4 {
            ctx.set_source_rgb(0., 0., 1.);
        } else if k == 3 {
            ctx.set_source_rgb(0., 0., 0.);
        }

        // ctx.move_to(tree[0][0][0] as f64, tree[0][0][1] as f64);
        // for edge in &tree {
        //     ctx.move_to(edge[0][0] as f64, edge[0][1] as f64);
        //     ctx.line_to(edge[1][0] as f64, edge[1][1] as f64);
        //     ctx.stroke();
        // }

        // for point in voronoi_points {
        //     ctx.move_to(point[0] as f64 - 0.5, point[1] as f64 - 0.5);
        //     ctx.line_to(point[0] as f64 + 0.5, point[1] as f64 + 0.5);
        //     ctx.stroke();
        // }

        if let Some(first) = tsp.first() {
            ctx.move_to(first[0] as f64, first[1] as f64);
        }
        for next in tsp.iter().skip(1) {
            ctx.line_to(next[0] as f64, next[1] as f64);
        }
        ctx.stroke();
    }

    Ok(())
}

pub fn abs_distance_squared<T: num_traits::int::PrimInt>(a: [T; 2], b: [T; 2]) -> T {
    let x_diff = if a[0] > b[0] {
        a[0] - b[0]
    } else {
        b[0] - a[0]
    };
    let y_diff = if a[1] > b[1] {
        a[1] - b[1]
    } else {
        b[1] - a[1]
    };
    x_diff.pow(2).saturating_add(y_diff.pow(2))
}
