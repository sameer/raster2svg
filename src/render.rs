use crate::{
    filter::{
        edge_flow_estimation, edge_tangent_flow, flow_based_difference_of_gaussians,
        step_edge_detection,
    },
    graph::{mst, tsp},
    voronoi::{calculate_cell_properties, colors_to_assignments, jump_flooding_voronoi},
    Style,
};
use cairo::{Context, Matrix};
use log::{debug, warn};
use lyon_geom::{point, Point};
use ndarray::prelude::*;
use spade::delaunay::IntDelaunayTriangulation;

pub fn render_fdog_based(
    image: ArrayView2<f64>,
    _super_sample: usize,
    instrument_diameter_in_pixels: f64,
    _style: Style,
    ctx: &Context,
    matrix: Matrix,
) {
    let mut image = image.to_owned();
    // Need to invert image for the sake of fdog
    image.par_mapv_inplace(|x| 1.0 - x);
    let etf = edge_tangent_flow(image.view());
    let fdog = flow_based_difference_of_gaussians(image.view(), etf.view());
    for (pos, value) in fdog.indexed_iter() {
        if *value < 1.0 {
            ctx.set_source_rgb(*value, *value, *value);
            ctx.set_matrix(matrix);
            ctx.move_to(pos.0 as f64, pos.1 as f64);
            ctx.rectangle(pos.0 as f64, pos.1 as f64, 1.0, 1.0);
            ctx.set_matrix(Matrix::identity());
            ctx.fill().unwrap();
        }
    }
}

/// Run Weighted Linde-Buzo-Gray Stippling and customize the output according to the desired style.
///
/// TODO: implement is assumed to be circular, can this support non-circular implements?
///
/// <http://graphics.uni-konstanz.de/publikationen/Deussen2017LindeBuzoGray/WeightedLindeBuzoGrayStippling_authorversion.pdf>
pub fn render_stipple_based(
    image: ArrayView2<f64>,
    super_sample: usize,
    implement_diameter_in_pixels: f64,
    style: Style,
    ctx: &Context,
    matrix: Matrix,
) {
    let (width, height) = image.dim();

    let stipple_area = (implement_diameter_in_pixels / 2.).powi(2) * std::f64::consts::PI;

    let mut voronoi_sites = vec![[(width / 2) as i64, (height / 2) as i64]];

    let initial_hysteresis = 0.6;
    let hysteresis_delta = 0.01;
    for iteration in 0..140 {
        if voronoi_sites.is_empty() {
            break;
        }
        debug!("Jump flooding voronoi");
        let colored_pixels = jump_flooding_voronoi(&voronoi_sites, width, height);
        let sites_to_points = colors_to_assignments(&voronoi_sites, colored_pixels.view());

        debug!("Linde-Buzo-Gray Stippling");

        let current_hysteresis = initial_hysteresis + iteration as f64 * hysteresis_delta;
        let remove_threshold = 1. - current_hysteresis / 2.;
        let split_threshold = 1. + current_hysteresis / 2.;

        let mut new_sites = Vec::with_capacity(voronoi_sites.len());
        let mut changed = false;
        for (_, points) in voronoi_sites.iter().zip(sites_to_points.iter()) {
            let cell_properties = calculate_cell_properties(image.view(), points);
            let moments = cell_properties.moments;

            // Density is very low, remove this point early
            if moments.m00 <= f64::EPSILON {
                changed = true;
                continue;
            }
            let centroid = cell_properties.centroid.unwrap();

            let scaled_density = moments.m00 / super_sample.pow(2) as f64;

            let line_area = stipple_area;

            let zero = Point::zero();
            let upper_bound = point((width - 1) as f64, (height - 1) as f64);

            if scaled_density < remove_threshold * line_area {
                changed = true;
                continue;
            } else if scaled_density < split_threshold * line_area {
                new_sites.push(
                    centroid
                        .clamp(zero, upper_bound)
                        .round()
                        .cast::<i64>()
                        .to_array(),
                );
            } else {
                if cell_properties
                    .phi_oriented_segment_through_centroid
                    .is_none()
                {
                    new_sites.push(
                        centroid
                            .clamp(zero, upper_bound)
                            .round()
                            .cast::<i64>()
                            .to_array(),
                    );
                    continue;
                }

                let line_segment = cell_properties
                    .phi_oriented_segment_through_centroid
                    .unwrap();
                let left = line_segment.sample(0.25);
                let right = line_segment.sample(0.75);

                if let Some((left, right)) = left
                    .round()
                    .clamp(zero, upper_bound)
                    .try_cast::<i64>()
                    .zip(right.round().clamp(zero, upper_bound).try_cast::<i64>())
                    .map(|(left, right)| (left.to_array(), right.to_array()))
                {
                    changed = true;
                    new_sites.push(left);
                    new_sites.push(right);
                } else {
                    warn!("could not split: {:?} {:?}", left, right);
                    new_sites.push(centroid.clamp(zero, upper_bound).cast::<i64>().to_array());
                }
            }
        }

        debug!(
            "Check stopping condition (iteration = {}, sites: {})",
            iteration,
            new_sites.len()
        );
        voronoi_sites = new_sites;
        if !changed {
            break;
        }
    }

    // On the off chance 2 sites end up being the same...
    voronoi_sites.sort_unstable();
    voronoi_sites.dedup();

    if voronoi_sites.len() < 3 {
        warn!(
            "Channel has too few vertices ({}) to draw, skipping",
            voronoi_sites.len()
        );
        return;
    }

    match style {
        Style::Stipple => {
            debug!("Draw to svg");
            for site in voronoi_sites {
                ctx.set_matrix(matrix);
                ctx.move_to(site[0] as f64, site[1] as f64);
                ctx.arc(
                    site[0] as f64,
                    site[1] as f64,
                    implement_diameter_in_pixels / 2.0,
                    0.,
                    std::f64::consts::TAU,
                );
                ctx.set_matrix(Matrix::identity());
                ctx.fill().unwrap();
            }
        }
        Style::Voronoi => {
            debug!("Draw to svg");
            let sites_to_points = colors_to_assignments(
                &voronoi_sites,
                jump_flooding_voronoi(&voronoi_sites, width, height).view(),
            );
            for points in sites_to_points {
                let properties = calculate_cell_properties(image, &points);
                if let Some(hull) = properties.hull {
                    ctx.set_matrix(matrix);
                    if let Some(first) = hull.first() {
                        ctx.move_to(first[0] as f64, first[1] as f64);
                    }
                    for point in hull.iter().skip(1) {
                        ctx.line_to(point[0] as f64, point[1] as f64);
                    }
                    ctx.set_matrix(Matrix::identity());
                    ctx.stroke().unwrap();
                }
            }
        }
        Style::Triangulation | Style::Mst | Style::Tsp => {
            let mut delaunay = IntDelaunayTriangulation::with_tree_locate();
            for vertex in &voronoi_sites {
                delaunay.insert([vertex[0] as i64, vertex[1] as i64]);
            }

            if let Style::Triangulation = style {
                debug!("Draw to svg");
                for edge in delaunay.edges() {
                    let from: &[i64; 2] = &edge.from();
                    let to: &[i64; 2] = &edge.to();

                    ctx.set_matrix(matrix);
                    ctx.move_to(from[0] as f64, from[1] as f64);
                    ctx.line_to(to[0] as f64, to[1] as f64);
                    ctx.set_matrix(Matrix::identity());
                    ctx.stroke().unwrap();
                }
            } else {
                let tree = mst::compute_mst(&voronoi_sites, &delaunay);
                if let Style::Mst = style {
                    debug!("Draw to svg");
                    ctx.set_matrix(matrix);
                    ctx.move_to(tree[0][0][0] as f64, tree[0][0][1] as f64);
                    ctx.set_matrix(Matrix::identity());
                    for edge in &tree {
                        ctx.set_matrix(matrix);
                        ctx.move_to(edge[0][0] as f64, edge[0][1] as f64);
                        ctx.line_to(edge[1][0] as f64, edge[1][1] as f64);
                        ctx.set_matrix(Matrix::identity());
                        ctx.stroke().unwrap();
                    }
                } else {
                    let tsp = tsp::approximate_tsp_with_mst(&voronoi_sites, &tree);
                    debug!("Draw to svg");
                    ctx.set_matrix(matrix);
                    if let Some(first) = tsp.first() {
                        ctx.move_to(first[0] as f64, first[1] as f64);
                    }
                    for next in tsp.iter().skip(1) {
                        ctx.line_to(next[0] as f64, next[1] as f64);
                    }
                    ctx.set_matrix(Matrix::identity());
                    ctx.stroke().unwrap();
                }
            }
        }
        Style::EdgesPlusHatching => unreachable!(),
    }
}
