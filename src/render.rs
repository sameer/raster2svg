use crate::{etf::edge_tangent_flow, Style};
use cairo::{Context, Matrix};
use log::{debug, warn};
use lyon_geom::{point, vector, LineSegment, Vector};
use ndarray::prelude::*;
use spade::delaunay::IntDelaunayTriangulation;

use crate::voronoi::{calculate_cell_properties, colors_to_assignments, jump_flooding_voronoi};

pub fn render_fdog_based(
    image: ArrayView2<f64>,
    super_sample: usize,
    instrument_diameter_in_pixels: f64,
    style: Style,
    ctx: &Context,
    matrix: Matrix,
) {
    let etf = edge_tangent_flow(image);
    for (pos, vector) in etf.indexed_iter() {
        ctx.set_line_width(0.1);
        ctx.set_matrix(matrix);
        ctx.move_to(pos.0 as f64, pos.1 as f64);
        ctx.rel_line_to(vector.x, vector.y);
        ctx.set_matrix(Matrix::identity());
        ctx.stroke();
    }
}

pub fn render_edge_based(
    image: ArrayView2<f64>,
    super_sample: usize,
    instrument_diameter_in_pixels: f64,
    style: Style,
    ctx: &Context,
    matrix: Matrix,
) {
    let [width, height] = if let [width, height] = *image.shape() {
        [width, height]
    } else {
        unreachable!()
    };
    let stipple_area = (instrument_diameter_in_pixels / 2.).powi(2) * std::f64::consts::PI;

    let etf = edge_tangent_flow(image);

    let zero = point(0., 0.);
    let upper_bound = point((width - 1) as f64, (height - 1) as f64);

    let mut voronoi_sites = vec![LineSegment {
        from: zero,
        to: upper_bound,
    }];
    let initial_hysteresis = 0.2;
    let hysteresis_delta = 0.01;
    for iteration in 0..50 {
        if voronoi_sites.is_empty() {
            break;
        }
        debug!("Jump flooding voronoi");
        let colored_pixels = jump_flooding_voronoi::<_, usize>(&voronoi_sites, width, height);
        let sites_to_points =
            colors_to_assignments::<_, usize>(&voronoi_sites, colored_pixels.view());

        debug!("Linde-Buzo-Gray Stippling");

        let current_hysteresis = initial_hysteresis + iteration as f64 * hysteresis_delta;
        let remove_threshold = 1. - current_hysteresis / 2.;
        let split_threshold = 1. + current_hysteresis / 2.;

        let mut new_sites = Vec::with_capacity(voronoi_sites.len());
        let mut changed = false;
        for (site, points) in voronoi_sites.iter().zip(sites_to_points.iter()) {
            let cell_properties = calculate_cell_properties(image.view(), points);
            let moments = cell_properties.moments;

            if moments.density <= f64::EPSILON {
                changed = true;
                continue;
            }
            let centroid = cell_properties.centroid.unwrap();

            let scaled_density = moments.density / super_sample.pow(2) as f64;

            let line_length = site.length();
            let line_area = ((line_length - instrument_diameter_in_pixels)
                * instrument_diameter_in_pixels
                + stipple_area)
                .max(stipple_area);

            if scaled_density < remove_threshold * line_area {
                changed = true;
                continue;
            } else if scaled_density < split_threshold * line_area {
                let mut acc = Vector::zero();
                for point in points {
                    acc += etf[*point];
                }
                let etf_vector = acc.normalize();
                let (sin, cos) = (-std::f64::consts::FRAC_PI_2).sin_cos();
                let flow_vector = vector(
                    etf_vector.x * cos - etf_vector.y * sin,
                    etf_vector.x * sin + etf_vector.y * cos,
                );
                new_sites.push(LineSegment {
                    from: (centroid - flow_vector * line_length * 0.5).clamp(zero, upper_bound),
                    to: (centroid + flow_vector * line_length * 0.5).clamp(zero, upper_bound),
                });
            } else {
                if cell_properties
                    .phi_oriented_segment_through_centroid
                    .is_none()
                {
                    new_sites.push(*site);
                    continue;
                }
                changed = true;
                let segment = cell_properties
                    .phi_oriented_segment_through_centroid
                    .unwrap();

                new_sites.push(LineSegment {
                    from: (segment.sample(0.6)).clamp(zero, upper_bound),
                    to: (segment.sample(0.8)).clamp(zero, upper_bound),
                });
                new_sites.push(LineSegment {
                    from: (segment.sample(0.2)).clamp(zero, upper_bound),
                    to: (segment.sample(0.4)).clamp(zero, upper_bound),
                });
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

    debug!("Draw to svg");
    for site in voronoi_sites {
        // let properties = calculate_cell_properties(image, &points);
        // if let Some(hull) = properties.hull {
        //     ctx.save();
        //     ctx.set_source_rgb(
        //         rng.gen_range(0.0..=1.0),
        //         rng.gen_range(0.0..=1.0),
        //         rng.gen_range(0.0..=1.0),
        //     );
        //     ctx.set_matrix(matrix);
        //     if let Some(first) = hull.first() {
        //         ctx.move_to(first[0] as f64, first[1] as f64);
        //     }
        //     for point in hull.iter().skip(1).chain(hull.first()) {
        //         ctx.line_to(point[0] as f64, point[1] as f64);
        //     }
        //     ctx.set_matrix(Matrix::identity());
        //     ctx.stroke();
        //     ctx.restore();
        // }
        ctx.set_matrix(matrix);
        ctx.move_to(site.from.x, site.from.y);
        ctx.line_to(site.to.x, site.to.y);
        ctx.set_matrix(Matrix::identity());
        ctx.stroke();
    }
}

/// Run Weighted Linde-Buzo-Gray Stippling and customize the output according to the desired style.
pub fn render_stipple_based(
    image: ArrayView2<f64>,
    super_sample: usize,
    instrument_diameter_in_pixels: f64,
    style: Style,
    ctx: &Context,
    matrix: Matrix,
) {
    let [width, height] = if let [width, height] = *image.shape() {
        [width, height]
    } else {
        unreachable!()
    };

    let stipple_area = (instrument_diameter_in_pixels / 2.).powi(2) * std::f64::consts::PI;

    let mut voronoi_sites = vec![[(width / 2) as i64, (height / 2) as i64]];

    let initial_hysteresis = 0.6;
    let hysteresis_delta = 0.01;
    for iteration in 0..50 {
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

            if moments.density == 0.0 {
                changed = true;
                continue;
            }
            let centroid = cell_properties.centroid.unwrap();

            let scaled_density = moments.density / super_sample.pow(2) as f64;

            let line_area = if matches!(style, Style::EdgeStipple) {
                if let Some(line_segment) = cell_properties
                    .phi_oriented_segment_through_centroid
                    .as_ref()
                {
                    ((line_segment.length() - instrument_diameter_in_pixels)
                        * instrument_diameter_in_pixels
                        + stipple_area)
                        .max(stipple_area)
                } else {
                    stipple_area
                }
            } else {
                stipple_area
            };

            let zero = point(0., 0.);
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
                    instrument_diameter_in_pixels / 2.0,
                    0.,
                    std::f64::consts::TAU,
                );
                ctx.set_matrix(Matrix::identity());
                ctx.fill();
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
                    ctx.stroke();
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
                    ctx.stroke();
                }
            } else {
                let tree = crate::mst::compute_mst(&voronoi_sites, &delaunay);
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
                        ctx.stroke();
                    }
                } else {
                    let tsp = crate::tsp::approximate_tsp_with_mst(&voronoi_sites, &tree);
                    debug!("Draw to svg");
                    ctx.set_matrix(matrix);
                    if let Some(first) = tsp.first() {
                        ctx.move_to(first[0] as f64, first[1] as f64);
                    }
                    for next in tsp.iter().skip(1) {
                        ctx.line_to(next[0] as f64, next[1] as f64);
                    }
                    ctx.set_matrix(Matrix::identity());
                    ctx.stroke();
                }
            }
        }
        Style::EdgeStipple => unreachable!(),
    }
}
