use ndarray::prelude::*;
use ndarray::{stack, Array1, Array2, Axis, Dim, Dimension};
use ndarray_stats::QuantileExt;

pub fn direct<F: Fn(&[f64]) -> f64>(cost_function: F, bounds: TwoD) {
    let mut state = State {
        min: None,
        evaluations: 0,
    };
    initialize(cost_function, bounds, &mut state);
}

struct State {
    min: Option<(OneD, f64)>,
    evaluations: usize,
}

type OneD = Array1<f64>;
type TwoD = Array2<f64>;

fn initialize<F: Fn(&[f64]) -> f64>(cost_function: F, bounds: TwoD, state: &mut State) {
    let dimensions = bounds.shape()[0];
    let center = Array::from_elem(dimensions, 0.5);
    let eval_center = cost_function(denormalize(center, bounds).as_slice().unwrap());

    let unit_bounds = stack![
        Axis(1),
        Array::<f64, _>::zeros(dimensions),
        Array::ones(dimensions)
    ];

    split(
        cost_function,
        state,
        Rectangle {
            bounds: unit_bounds,
            eval: eval_center,
        },
    )
}

fn denormalize(point: OneD, bounds: TwoD) -> Array<f64, Dim<[usize; 1]>> {
    let bounds_range = bounds.slice(s![.., 1]).to_owned() - bounds.slice(s![.., 0]);
    let bounds_min = bounds.slice(s![.., 0]);
    point * bounds_range + bounds_min
}

struct Rectangle {
    bounds: TwoD,
    eval: f64,
}

fn split<F: Fn(&[f64]) -> f64>(cost_function: F, state: &mut State, rectangle: Rectangle) {
    let bounds_range =
        rectangle.bounds.slice(s![.., 1]).to_owned() - rectangle.bounds.slice(s![.., 0]);
    let dimensions = bounds_range.shape()[0];
    let is_cube =
        bounds_range.abs_diff_eq(&Array::from_elem(dimensions, bounds_range[0]), f64::EPSILON);

    let (splitting_offset, dei) = if is_cube {
        (0usize, Array2::from_diag(&(bounds_range.clone() / 3.)))
    } else {
        let splitting_offset = bounds_range.argmax().unwrap();
        let mut dei = Array::<f64, _>::zeros((1, dimensions));
        dei[[0, splitting_offset]] = bounds_range[splitting_offset] / 3.;
        (splitting_offset, dei)
    };

    let center = rectangle.bounds.slice(s![.., 0]).to_owned() + bounds_range / 2.;

    let center_dei = stack![Axis(1), center.clone() - &dei, center + &dei];
    let eval_center_dei = Array::from_vec(
        center_dei
            .lanes(Axis(2))
            .into_iter()
            .map(|x| cost_function(x.as_slice().unwrap()))
            .collect::<Vec<f64>>(),
    );
    state.evaluations += eval_center_dei.len();

    let min_idx = eval_center_dei.argmin().unwrap();
    let eval_center_dei_min = eval_center_dei[min_idx];
    let eval_center_dei = eval_center_dei
        .into_shape((center_dei.shape()[0], center_dei.shape()[1]))
        .unwrap();
    if let Some(min) = &mut state.min {
        if eval_center_dei_min < min.1 {
            *min = (
                center_dei
                    .lanes(Axis(2))
                    .into_iter()
                    .nth(min_idx)
                    .unwrap()
                    .to_owned(),
                eval_center_dei_min,
            );
        }
    } else {
        state.min = Some((
            center_dei
                .lanes(Axis(2))
                .into_iter()
                .nth(min_idx)
                .unwrap()
                .to_owned(),
            eval_center_dei_min,
        ));
    }

    let wi_indices = eval_center_dei.argmin(eval_center_dei, );
}
#[test]
fn test() {
    direct(|point| point[0] + point[1], array!([0., 25.], [0., 10.]));
}
