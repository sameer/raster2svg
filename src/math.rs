use num_traits::{PrimInt, Signed};
use std::fmt::Debug;

#[macro_export]
macro_rules! kbn_summation {
    (for $pat: pat in $expr: expr => {
        $('loop: { $(let $loopvar: ident = $loopvar_expr: expr;)* })?
        $($var: ident += $var_expr: expr;)*
    }) => {
        let ($($var,)*) = {
            use paste::paste;
            paste! {
                $(
                    let mut $var: f64 = 0.;
                    let mut [<$var compensation>] = 0.;
                )*
                    for $pat in $expr {
                        $($(let $loopvar = $loopvar_expr;)*)?
                        $(
                            let input = $var_expr;
                            let t = $var + input;
                            [<$var compensation>] += if $var.abs() >= input.abs() {
                                ($var - t) + input
                            } else {
                                (input - t) + $var
                            };
                            $var = t;
                        )*
                    }
                ($($var + [<$var compensation>],)*)
            }
        };
    };
}

/// Square of the Euclidean distance between signed 2D coordinates
#[inline]
pub fn abs_distance_squared<T: PrimInt + Signed + Debug>(a: [T; 2], b: [T; 2]) -> T {
    let x_diff = a[0] - b[0];
    let y_diff = a[1] - b[1];
    debug_assert!(
        x_diff.pow(2).checked_add(&y_diff.pow(2)).is_some(),
        "x_diff = {:?}, y_diff = {:?}",
        x_diff,
        y_diff
    );
    x_diff.pow(2) + y_diff.pow(2)
}

#[cfg(test)]
#[test]
fn test_summation() {
    use std::f64::consts::*;
    let input = [FRAC_PI_8, FRAC_PI_2, FRAC_PI_6, FRAC_PI_3, FRAC_PI_4];
    kbn_summation! {
        for x in input => {
            out += x;
        }
    }

    assert_ne!(input.iter().sum::<f64>(), out);
    assert_eq!(out, 4.31968989868596570288)
}
