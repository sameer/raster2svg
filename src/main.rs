use cairo::Context;
use image::{io::Reader as ImageReader, ImageBuffer, Pixel, Rgb};
use log::*;
use ndarray::prelude::*;

use rand::prelude::*;
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

fn main() -> io::Result<()> {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "raster2svg=info")
    }
    env_logger::init();
    let opt = Opt::from_args();

    let image = {
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
        .to_rgb8();

        Array::from_iter(image.pixels().map(|p| p.0.iter().copied()).flatten())
            .into_shape((image.height() as usize, image.width() as usize, 3))
            .unwrap()
            .reversed_axes()
    };

    // https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
    let dither = {
        let mut dither = image.clone();
        for k in 0..3 {
            for j in 0..dither.shape()[2] {
                for i in 0..dither.shape()[1] {
                    let original_value = dither[[k, i, j]];
                    let new_value = if original_value >= 128 {
                        u8::MAX
                    } else {
                        u8::MIN
                    };
                    dither[[k, i, j]] = new_value;
                    const OFFSETS: [[isize; 2]; 4] = [[1, 0], [-1, 1], [0, 1], [1, 1]];
                    const QUANTIZATION: [u16; 4] = [7, 3, 5, 1];
                    let (errs, add) = if original_value > new_value {
                        let mut quantization_errors = [0; 4];
                        for (idx, q) in QUANTIZATION.iter().enumerate() {
                            quantization_errors[idx] =
                                ((q * ((original_value - new_value) as u16)) / 16) as u8;
                        }
                        (quantization_errors, true)
                    } else {
                        let mut quantization_errors = [0; 4];
                        for (idx, q) in QUANTIZATION.iter().enumerate() {
                            quantization_errors[idx] =
                                ((q * ((new_value - original_value) as u16)) / 16) as u8;
                        }
                        (quantization_errors, false)
                    };
                    for (offset, err) in OFFSETS.iter().zip(errs.iter()) {
                        let index = [
                            k as isize,
                            (i as isize + offset[0]).clamp(-1, (dither.shape()[1] - 1) as isize),
                            (j as isize + offset[1]).clamp(-1, (dither.shape()[2] - 1) as isize),
                        ];
                        let value = if add {
                            dither
                                .slice(s![index[0], index[1], index[2]])
                                .into_scalar()
                                .saturating_add(*err)
                        } else {
                            dither
                                .slice(s![index[0], index[1], index[2]])
                                .into_scalar()
                                .saturating_sub(*err)
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

    // TODO: handle black pen
    // let key = dither.map_axis(Axis(0), |pixel| {
    //     (((pixel[0] as u16 + pixel[1] as u16 + pixel[2] as u16) / (3 * 255)) / 255) as u8
    // });
    let pen_diameter = opt.pen_diameter.get::<millimeter>();
    let in_to_mm = Length::new::<inch>(1.).get::<millimeter>();
    let width = (image.shape()[1] as f64 / opt.dots_per_inch * in_to_mm).round();
    let height = (image.shape()[2] as f64 / opt.dots_per_inch * in_to_mm).round();

    let mut surf = match &opt.out {
        Some(_) => cairo::SvgSurface::new(width, height, opt.out).unwrap(),
        None => cairo::SvgSurface::for_stream(width, height, std::io::stdout()).unwrap(),
    };
    surf.set_document_unit(cairo::SvgUnit::Mm);
    let ctx = Context::new(&surf);

    ctx.set_source_rgb(0., 0., 0.);
    ctx.rectangle(0., 0., width, height);
    ctx.fill();

    ctx.set_line_width(pen_diameter);
    // Makes life easier to work with the same coords as the image
    ctx.scale(
        width / image.shape()[1] as f64,
        height / image.shape()[2] as f64,
    );

    let mut rng = thread_rng();
    for k in 0..3 {
        let mut voronoi_vertices = {
            let mut indices = dither
                .slice(s![k, .., ..])
                .indexed_iter()
                .filter_map(|idx_and_elem: ((usize, usize), &u8)| {
                    if *idx_and_elem.1 == u8::MAX {
                        Some([idx_and_elem.0 .0 as i64, idx_and_elem.0 .1 as i64])
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            indices.shuffle(&mut rng);
            indices.truncate(indices.len().min(10000));
            indices
            // let width = Uniform::new(0, image.shape()[1]);
            // let height = Uniform::new(0, image.shape()[2]);
            // (0..10000)
            //     .map(|_| (rng.sample(width), rng.sample(height)))
            //     .collect::<Vec<_>>()
        };
        // let mut point_assignments;

        // loop {
        // debug!("Naive voronoi");
        // point_assignments =
        //     voronoi::compute_voronoi(&voronoi_vertices, image.shape()[1], image.shape()[2]);

        // break;
        // debug!("Lloyd's algorithm");
        // let centroids: Vec<(usize, usize)> = point_assignments
        //     .iter()
        //     .map(|point_assignment| {
        //         let denominator = point_assignment.iter().fold(0., |mut acc, (x, y)| {
        //             acc += image[[k, *x, *y]] as f64 / 255.;
        //             acc
        //         });
        //         let numerator_y = point_assignment.iter().fold(0., |mut acc, (x, y)| {
        //             acc += *y as f64 * image[[k, *x, *y]] as f64 / 255.;
        //             acc
        //         });
        //         let numerator_x = point_assignment.iter().fold(0., |mut acc, (x, y)| {
        //             acc += *x as f64 * (image[[k, *x, *y]] as f64 / 255.);
        //             acc
        //         });
        //         (numerator_x / denominator, numerator_y / denominator)
        //     })
        //     .map(|(x, y)| (x.round() as usize, y.round() as usize))
        //     .collect();

        // debug!("Check stopping condition");
        // // stopping condition: average distance moved by all stippling points is small
        // let distance = voronoi_vertices
        //     .iter()
        //     .zip(centroids.iter())
        //     .map(|(voronoi, centroid)| abs_distance_squared(*voronoi, *centroid))
        //     .fold(0, |acc, dist| acc + dist);
        // voronoi_vertices = centroids;
        // let average_distance = (distance as f64).sqrt() / voronoi_vertices.len() as f64;
        // debug!("at {}", average_distance);
        // if average_distance <= 0.0001 {
        //     debug!("done with color channel {}", k);
        //     break;
        // }
        // }

        if k == 0 {
            ctx.set_source_rgb(1., 0., 0.);
        } else if k == 1 {
            ctx.set_source_rgb(0., 1., 0.);
        } else if k == 2 {
            ctx.set_source_rgb(0., 0., 1.);
        }

        // let mut visited = vec![false; voronoi_points.len()];
        // let mut path: Vec<(usize, usize)> = Vec::with_capacity(voronoi_points.len());
        // debug!("Approximate tsp path");
        // let path = tsp::approximate_tsp_with_mst(&voronoi_vertices);

        let mut delaunay = IntDelaunayTriangulation::with_tree_locate();
        for vertex in &voronoi_vertices {
            delaunay.insert(*vertex);
        }

        let tree = crate::mst::compute_mst(&voronoi_vertices, &delaunay);
        let tsp = crate::tsp::approximate_tsp_with_mst(&voronoi_vertices, &tree);
        debug!("Draw to svg");
        // if k == 1 {
        // for face in delaunay.triangles() {
        //     let triangle = face.as_triangle();
        //     ctx.move_to(triangle[0][0] as f64, triangle[0][1] as f64);
        //     ctx.line_to(triangle[1][0] as f64, triangle[1][1] as f64);
        //     ctx.line_to(triangle[2][0] as f64, triangle[2][1] as f64);
        //     ctx.stroke();
        // }
        ctx.move_to(tsp[0][0][0] as f64, tsp[0][0][1] as f64);
        for edge in &tsp {
            ctx.line_to(edge[1][0] as f64, edge[1][1] as f64);
        }
        ctx.stroke();
        // }
        // if let Some(edge) = path.first() {
        //     ctx.move_to(
        //         voronoi_vertices[edge.0].0 as f64,
        //         voronoi_vertices[edge.0].1 as f64,
        //     );
        // }
        // for edge in path {
        //     let vertex = voronoi_vertices[edge.1];
        //     ctx.line_to(vertex.0 as f64, vertex.1 as f64);
        // }
        // ctx.stroke();
    }
    // let out = ImageBuffer::from_fn(image.shape()[1] as u32, image.shape()[2] as u32, |x, y| {
    //     let x = x as usize;
    //     let y = y as usize;
    //     Rgb([dither[[0, x, y]], dither[[1, x, y]], dither[[2, x, y]]])
    // });
    // out.save_with_format("out.png", image::ImageFormat::Png)
    //     .unwrap();

    Ok(())
}

pub fn abs_distance_squared(a: [i64; 2], b: [i64; 2]) -> i64 {
    fn abs_difference<T: std::ops::Sub<Output = T> + Ord>(x: T, y: T) -> T {
        if x < y {
            y - x
        } else {
            x - y
        }
    }
    abs_difference(a[0], b[0]).pow(2) + abs_difference(a[1], b[1]).pow(2)
}
