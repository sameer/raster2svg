use crate::abs_distance_squared;
use fxhash::{FxHashMap as HashMap, FxHashSet as HashSet};
use rand::{distributions::Standard, prelude::Distribution, thread_rng, Rng};

#[derive(PartialEq, Eq, Debug)]
struct Edge([[i64; 2]; 2]);

impl Edge {
    fn length(&self) -> i64 {
        abs_distance_squared(self.0[0], self.0[1])
    }
}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.length().partial_cmp(&other.length())
    }
}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.length().cmp(&other.length())
    }
}

#[derive(PartialEq, Eq, Debug)]
struct RemoveAddEdgePair {
    original_edge: [[i64; 2]; 2],
    new_edge: [[i64; 2]; 2],
}

impl RemoveAddEdgePair {
    /// Best improvement to the tree (greatest reduction or smallest increase in length)
    fn diff(&self) -> i64 {
        abs_distance_squared(self.original_edge[0], self.original_edge[1])
            - abs_distance_squared(self.new_edge[0], self.new_edge[1])
    }
}

impl PartialOrd for RemoveAddEdgePair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.diff().partial_cmp(&other.diff())
    }
}

impl Ord for RemoveAddEdgePair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.diff().cmp(&other.diff())
    }
}

/// Approximate an open loop TSP solution by way of greedy branch elimination + local improvement.
///
/// http://cs.uef.fi/sipu/pub/applsci-11-00177.pdf
///
/// See the two source code comments prefixed with `NOTE:`
/// for how to use the non-greedy all pairs branch elimination
/// variant, which takes significantly longer.
///
/// TODO: consider using the Christofides algorithm which has a bound on the length of the worst path.
/// The initial solution of greedy branch elimination is actually quite bad.
pub fn approximate_tsp_with_mst(vertices: &[[i64; 2]], tree: &[[[i64; 2]; 2]]) -> Vec<[i64; 2]> {
    if vertices.len() <= 2 {
        return vec![];
    }
    let mut adjacency_map: HashMap<[i64; 2], HashSet<[i64; 2]>> = HashMap::default();
    tree.iter().for_each(|edge| {
        adjacency_map.entry(edge[0]).or_default().insert(edge[1]);
        adjacency_map.entry(edge[1]).or_default().insert(edge[0]);
    });

    let vertex_to_index = vertices
        .iter()
        .enumerate()
        .map(|(i, vertex)| (*vertex, i))
        .collect::<HashMap<_, _>>();
    let mut branch_list = adjacency_map
        .iter()
        .filter(|(_, adjacencies)| adjacencies.len() >= 3)
        .map(|(branch, adjacencies)| {
            adjacencies
                .iter()
                .map(move |adjacency| Edge([*branch, *adjacency]))
        })
        .flatten()
        .collect::<Vec<_>>();
    branch_list.sort();

    while let Some(Edge([branch, disconnected_node])) = branch_list.pop() {
        // No longer a branch
        if adjacency_map.get(&branch).unwrap().len() <= 2 {
            continue;
        }
        // Disconnected node was once a branch and already processed this pair
        if !adjacency_map
            .get(&branch)
            .unwrap()
            .contains(&disconnected_node)
        {
            continue;
        }

        // Remove edge
        adjacency_map
            .get_mut(&branch)
            .unwrap()
            .remove(&disconnected_node);
        adjacency_map
            .get_mut(&disconnected_node)
            .unwrap()
            .remove(&branch);

        // Now there are (in theory) two disconnected trees
        // Find the two connected trees in the graph
        let mut branch_tree_visited: Vec<bool> = vec![false; vertices.len()];
        {
            branch_tree_visited[*vertex_to_index.get(&branch).unwrap()] = true;
            let mut branch_dfs = vec![&branch];
            while let Some(head) = branch_dfs.pop() {
                for adjacency in adjacency_map.get(head).unwrap() {
                    let adjacency_idx = *vertex_to_index.get(adjacency).unwrap();
                    // // Explicitly skip this node to avoid entering the other connected component
                    // assert_ne!(*adjacency, disconnected_node);
                    // if *adjacency == disconnected_node {

                    // } else
                    if !branch_tree_visited[adjacency_idx] {
                        branch_tree_visited[adjacency_idx] = true;
                        branch_dfs.push(adjacency);
                    }
                }
            }
        }

        // Find leaves in the two
        // Pick the shortest possible link between two leaves that would reconnect the trees

        let (branch_tree_leaves, disconnected_tree_leaves) = branch_tree_visited
            .iter()
            .enumerate()
            .filter(|(i, _)| adjacency_map.get(&vertices[*i]).unwrap().len() <= 1)
            .partition::<Vec<_>, _>(|(_, in_branch_tree)| **in_branch_tree);

        let (branch_tree_leaf, disconnected_tree_leaf) = branch_tree_leaves
            .iter()
            .map(|(branch_idx, _)| vertices[*branch_idx])
            .map(|branch_tree_leaf| {
                disconnected_tree_leaves
                    .iter()
                    .map(|(disconnected_idx, _)| vertices[*disconnected_idx])
                    .map(move |disconnected_tree_leaf| (branch_tree_leaf, disconnected_tree_leaf))
            })
            .flatten()
            .min_by_key(|(branch_tree_leaf, disconnected_tree_leaf)| {
                abs_distance_squared(*branch_tree_leaf, *disconnected_tree_leaf)
            })
            .unwrap();

        // Connect leaves
        adjacency_map
            .get_mut(&branch_tree_leaf)
            .unwrap()
            .insert(disconnected_tree_leaf);
        adjacency_map
            .get_mut(&disconnected_tree_leaf)
            .unwrap()
            .insert(branch_tree_leaf);
    }

    // Extract path from the adjacency list
    let mut path: Vec<[i64; 2]> = Vec::with_capacity(vertices.len().saturating_sub(1));
    if let Some((first_vertex, adjacencies)) = adjacency_map
        .iter()
        .find(|(_, adjacencies)| adjacencies.len() == 1)
    {
        path.push(*first_vertex);
        path.push(*adjacencies.iter().next().unwrap());
    }

    // The number of edges in an open loop TSP path is equal to the number of vertices - 1
    while path.len() < vertices.len() {
        let last_vertex = *path.last().unwrap();
        let second_to_last_vertex = *path.iter().rev().skip(1).next().unwrap();
        let next_vertex = adjacency_map
            .get(&last_vertex)
            .unwrap()
            .iter()
            .find(|adjacency| **adjacency != second_to_last_vertex)
            .unwrap();
        path.push(*next_vertex);
    }
    local_improvement(&mut path)
}

#[derive(Debug, PartialEq, Eq)]
enum Operator {
    /// Move vertex between other vertices
    Relocate,
    /// Swap the positions of two vertices
    ///
    /// As the name suggests, it tends to
    /// remove crossed edges in the path.
    ///
    /// This is by my own design based on intuition around
    /// good vs bad tours generated by the greedy approximation.
    Disentangle,
    /// Swap vertices between two edges
    TwoOpt,
    /// Change the beginning and/or end of the path by swapping an edge
    LinkSwap,
}

impl Distribution<Operator> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Operator {
        match rng.gen_range(0..=2) {
            0 => Operator::Relocate,
            1 => Operator::Disentangle,
            2 => Operator::TwoOpt,
            _ => Operator::LinkSwap,
        }
    }
}

/// Local improvement of an open loop TSP solution using the relocate, disentangle, 2-opt, and link swap operators.
///
/// https://www.mdpi.com/2076-3417/9/19/3985/pdf
///
fn local_improvement(path: &[[i64; 2]]) -> Vec<[i64; 2]> {
    let mut best = path.to_owned();
    let mut best_sum = best
        .iter()
        .zip(best.iter().skip(1))
        .map(|(from, to)| abs_distance_squared(*from, *to))
        .sum::<i64>();

    let mut current = best.clone();
    let mut simulated_annealing_noise_min = 0.;
    let mut rng = thread_rng();

    let sample_count = (current.len().pow(2) as f64 * 0.001) as usize;
    const ITERATIONS: usize = 10000;
    for idx in ITERATIONS / 2..ITERATIONS {
        let edge_iterator = (0..current.len()).zip(1..current.len());
        let operator: Operator = rng.gen();

        // TODO: try using disentanglement as a metric as opposed to distance
        match operator {
            Operator::Relocate => {}
            Operator::Disentangle => {
                let mut swaps = (0..sample_count)
                    .map(|_| {
                        let i = rng.gen_range(0..current.len());
                        let j = if (i.saturating_add(2) >= current.len())
                            || (i > 1 && rng.gen::<bool>())
                        {
                            rng.gen_range(0..i.saturating_sub(1))
                        } else {
                            rng.gen_range(i + 2..current.len())
                        };
                        if i < j {
                            (i, j)
                        } else {
                            (j, i)
                        }
                    })
                    .collect::<Vec<_>>();
                if let Some((i, j, diff)) = swaps
                    .drain(..)
                    .map(|(i, j)| {
                        let mut diff: i64 = 0;
                        // TODO: fix bug in swap diff calculation
                        if i != 0 {
                            diff += abs_distance_squared(current[i - 1], current[i])
                                - abs_distance_squared(current[i - 1], current[j]);
                        }
                        diff += abs_distance_squared(current[i], current[i + 1])
                            - abs_distance_squared(current[j], current[i + 1]);
                        diff += abs_distance_squared(current[j - 1], current[j])
                            - abs_distance_squared(current[j - 1], current[i]);
                        if j + 1 != current.len() {
                            diff += abs_distance_squared(current[j], current[j + 1])
                                - abs_distance_squared(current[i], current[j + 1]);
                        }
                        (i, j, diff)
                    })
                    .max_by_key(|(_, _, diff)| {
                        if idx < ITERATIONS / 2 {
                            (*diff as f64 * rng.gen_range(simulated_annealing_noise_min..=1.))
                                as i64
                        } else {
                            *diff
                        }
                    })
                {
                    if diff <= 0 {
                        continue;
                    }

                    current.swap(i, j);
                    let current_sum = current
                        .iter()
                        .zip(current.iter().skip(1))
                        .map(|(from, to)| abs_distance_squared(*from, *to))
                        .sum::<i64>();
                    dbg!(idx, current_sum, best_sum);
                    if current_sum < best_sum {
                        best = current.clone();
                        best_sum = current_sum;
                    }
                }
            }
            Operator::TwoOpt => {
                let mut edge_pairs = (0..sample_count)
                    .map(|_| {
                        let i = rng.gen_range(0..current.len().saturating_sub(3));
                        let j = i + 1;
                        let other_i =
                            rng.gen_range(j.saturating_add(1)..current.len().saturating_sub(1));
                        let other_j = other_i + 1;
                        ((i, j), (other_i, other_j))
                    })
                    .collect::<Vec<_>>();
                if let Some((this, other, this_pair, other_pair)) = edge_pairs
                    .drain(..)
                    // = edge_iterator
                    //     .map(|(i, j)| {
                    //         let other_edge_iterator = j.saturating_add(1)..current.len();
                    //         other_edge_iterator
                    //             .clone()
                    //             .zip(other_edge_iterator.clone().skip(1))
                    //             .map(move |other| ((i, j), other))
                    //     })
                    //     .flatten()
                    .map(|(this, other)| {
                        (
                            this,
                            other,
                            RemoveAddEdgePair {
                                original_edge: [current[this.0], current[this.1]],
                                new_edge: [current[this.0], current[other.0]],
                            },
                            RemoveAddEdgePair {
                                original_edge: [current[other.0], current[other.1]],
                                new_edge: [current[this.1], current[other.1]],
                            },
                        )
                    })
                    .max_by_key(|(_, _, this_pair, other_pair)| {
                        if idx < ITERATIONS / 2 {
                            ((this_pair.diff() + other_pair.diff()) as f64
                                * rng.gen_range(simulated_annealing_noise_min..=1.))
                                as i64
                        } else {
                            this_pair.diff() + other_pair.diff()
                        }
                    })
                {
                    let diff = this_pair.diff() + other_pair.diff();
                    if diff <= 0 {
                        continue;
                    }

                    let reversed_middle = current[this.1..=other.0]
                        .iter()
                        .rev()
                        .copied()
                        .collect::<Vec<_>>();
                    current[this.1..=other.0]
                        .iter_mut()
                        .zip(reversed_middle.iter())
                        .for_each(|(dest, origin)| *dest = *origin);

                    let current_sum = current
                        .iter()
                        .zip(current.iter().skip(1))
                        .map(|(from, to)| abs_distance_squared(*from, *to))
                        .sum::<i64>();
                    dbg!(idx, current_sum, best_sum);

                    if current_sum < best_sum {
                        best = current.clone();
                        best_sum = current_sum;
                    }
                }
            }
            Operator::LinkSwap => {
                let first = *current.first().unwrap();
                let last = *current.last().unwrap();
                if let Some((i, j, pair, diff)) = edge_iterator
                    .filter_map(|(i, j)| {
                        let from = current[i];
                        let to = current[j];
                        [[from, last], [first, to], [first, last]]
                            .iter()
                            .map(|new_edge| {
                                let pair = RemoveAddEdgePair {
                                    new_edge: *new_edge,
                                    original_edge: [from, to],
                                };
                                let diff = pair.diff();
                                (i, j, pair, diff)
                            })
                            .max_by_key(|(_, _, _, diff)| *diff)
                    })
                    .max_by_key(|(_, _, _, diff)| {
                        if idx < ITERATIONS / 2 {
                            (*diff as f64 * rng.gen_range(simulated_annealing_noise_min..=1.))
                                as i64
                        } else {
                            *diff
                        }
                    })
                {
                    if diff <= 0 {
                        continue;
                    }

                    if pair.new_edge[0] != pair.original_edge[0] {
                        let reversed_head = current[..=i].iter().rev().copied().collect::<Vec<_>>();
                        current[..=i]
                            .iter_mut()
                            .zip(reversed_head.iter())
                            .for_each(|(dest, origin)| *dest = *origin);
                    }
                    if pair.new_edge[1] != pair.original_edge[1] {
                        let reversed_tail = current[j..].iter().rev().copied().collect::<Vec<_>>();
                        current[j..]
                            .iter_mut()
                            .zip(reversed_tail.iter())
                            .for_each(|(dest, origin)| *dest = *origin);
                    }

                    let current_sum = current
                        .iter()
                        .zip(current.iter().skip(1))
                        .map(|(from, to)| abs_distance_squared(*from, *to))
                        .sum::<i64>();
                    dbg!(idx, diff, current_sum, best_sum);

                    if current_sum < best_sum {
                        best = current.clone();
                        best_sum = current_sum;
                    }
                }
            }
        }
        simulated_annealing_noise_min =
            (simulated_annealing_noise_min + 2. / ITERATIONS as f64).clamp(0., 1.);
    }

    best
}

#[cfg(test)]
#[test]
fn tsp_is_correct_for_trivial_case() {
    let vertices = [[0, 0], [1, 1], [2, 2]];
    let tree = [[[0, 0], [1, 1]], [[1, 1], [2, 2]]];

    let path = approximate_tsp_with_mst(&vertices, &tree);
    let length: i64 = path
        .iter()
        .zip(path.iter().skip(1))
        .map(|(from, to)| abs_distance_squared(*from, *to))
        .sum();
    assert_eq!(length, 4);
}

#[cfg(test)]
#[test]
fn tsp_is_correct_for_nontrivial_case() {
    let vertices: [[i64; 2]; 5] = [[24, 28], [371, 33], [235, 72], [157, 509], [156, 194]];
    let tree = [
        [[24, 28], [156, 194]],
        [[156, 194], [235, 72]],
        [[235, 72], [371, 33]],
        [[156, 194], [157, 509]],
    ];
    let path = approximate_tsp_with_mst(&vertices, &tree);
    let length: i64 = path
        .iter()
        .zip(path.iter().skip(1))
        .map(|(from, to)| abs_distance_squared(*from, *to))
        .sum();
    assert_eq!(length, 210680);
    assert_eq!(&path, &vertices);
}
