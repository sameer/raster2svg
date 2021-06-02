# raster2svg

Convert raster graphics into stylistic art renderings. The output of this program is intended for direct use with [svg2gcode](https://github.com/sameer/svg2gcode) to draw with a pen plotter.

## Pipeline

1. Image pre-processing
    * Derive key (black) using D50 illumination lightness (D65 sRGB to D50 CIELAB)
    * Square pixel values and do error diffusion
1. Estimate good stipple counts per-channel using Floyd-Steinberg dither
1. Use Secord's algorithm to find stippling points
    * Find Voronoi tesselation with Jump flooding algorithm (JFA)
    * Calculate cell centroids and move points to them
    * Repeat until convergence
1. Get Delaunay triangulation using [spade](https://github.com/Stoeoef/spade)
1. Find Euclidean MST with Prim's algorithm using edges from the Delaunay triangulation
1. Approximate an open-loop TSP path through the points
    * MST to TSP
    * Local Improvement with 4 operators: relocate, disentangle, 2-opt, and link swap
1. Draw to SVG


# References

In no particular order, all of these papers/links provided general guidance in creating raster2svg.

* Weighted Voronoi Stippling https://www.cs.ubc.ca/labs/imager/tr/2002/secord2002b/secord.2002b.pdf
* Fast Capacity Constrained Voronoi Tessellation https://www.microsoft.com/en-us/research/wp-content/uploads/2009/10/paper-1.pdf
* Weighted Linde-Buzo-Gray Stippling http://graphics.uni-konstanz.de/publikationen/Deussen2017LindeBuzoGray/WeightedLindeBuzoGrayStippling_authorversion.pdf
* Halftoning and Stippling http://graphics.uni-konstanz.de/publikationen/Deussen2013HalftoningStippling/Deussen2013HalftoningStippling.pdf
* Capacity-constrained point distributions https://sci-hub.st/https://doi.org/10.1145/1576246.1531392
* TSP Art https://archive.bridgesmathart.org/2005/bridges2005-301.pdf
* Opt Art: Special Cases https://www.semanticscholar.org/paper/Opt-Art%3A-Special-Cases-Bosch/532945da714768a106096cf2537293a012898a0e
* Linking soft computing to art introduction of efficient k-continuous line drawing https://ieyjzhou.github.io/CIEG/Paper/KCLD_2018_Published_Version.pdf
* From Stippling to Scribbling http://archive.bridgesmathart.org/2015/bridges2015-267.pdf
* Converting MST to TSP Path by Branch Elimination http://cs.uef.fi/sipu/pub/applsci-11-00177.pdf
* Which Local Search Operator Works Best for the Open-Loop TSP? https://www.mdpi.com/2076-3417/9/19/3985/pdf
* Algorithms for computing EMSTs in two dimensions https://en.wikipedia.org/wiki/Euclidean_minimum_spanning_tree#Algorithms_for_computing_EMSTs_in_two_dimensions
* RGB/XYZ Matrices http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html

There are also some noteworthy papers that while interesting did not influence raster2svg.

* Amplitude Modulated Line-Based Halftoning http://graphics.uni-konstanz.de/publikationen/Ahmed2016AmplitudeModulatedLine/paper.pdf
* Structure grid for directional stippling http://www.cs.umsl.edu/~kang/Papers/kang_gm2011.pdf