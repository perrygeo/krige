// TODO
// calculate optimal raster grid (affine, shape) based on bounding box
// fit model automaticaly
//   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy-optimize-curve-fit
// support other models: Exponential, Linear, Gaussian
//   https://mmaelicke.github.io/scikit-gstat/reference/models.html
// cross-validation, MSPE on a subset of the original (removed prior to constructing empirical semivariogram)
//
// packaging:
// python interface
// make an API and write unit tests
// error handling
//
// outputs:
// interpolate over the grid in parallel, returning 2D array + affine (for rasterio to reconstruct a raster)
// interpolate over hex grid or user-supplied points, returning a 1D array per feature
//
// variogram:
// geodesic distance, great circle distance, ECEF 3D distance
// Co-kriging
// Trend elimination (UK)
// Subsetting
// Anisotropy
// Bayesian estimation

use itertools::Itertools;
use std::error::Error;
use std::fs::File;

use clap::Parser;
use kdtree::{distance::squared_euclidean, KdTree};
use rand::{seq::IteratorRandom, thread_rng};

use rust_interpolate::{
    create_matrix_a, create_vector_b, empirical_semivariogram, fit_model, parse_locations, predict,
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Filename of the XYZ tab delimited text file
    #[clap(short, long)]
    points: String,

    /// Maxiumum Range or Lag distance to consider
    #[clap(short, long)]
    range: f64,

    /// Number of locations to sample for the empirical semivariogram
    #[clap(short, long, default_value_t = 5000)]
    samples: usize,

    /// Number of lag bins for the empirical semivariogram
    #[clap(short, long, default_value_t = 256)]
    nbins: usize,

    /// Max Number of neighboring data points to consider
    #[clap(short, long, default_value_t = 64)]
    max_neighbors: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let mut rng = thread_rng();

    // Parse XYZ Locations from input text
    eprintln!("Parse XYZ file...");
    let file = File::open(args.points)?;
    let locs = parse_locations(file)?;

    // -------- Create spatial index, full dataset
    eprintln!("Create spatial index and removing dups...");
    const EPSILON: f64 = 1e-9 * 1e-9;
    let mut kdtree = KdTree::new(2);
    for loc in locs.iter() {
        // If existing points, check to avoid dups
        // there must be a better way!
        let nns = kdtree.nearest(&[loc.x, loc.y], 1, &squared_euclidean)?;
        if nns.len() > 0 {
            let (sqdist, _existing) = nns[0];
            if sqdist < EPSILON {
                // Duplicate point exists, skipping
                continue;
            }
        }
        // New unique point
        kdtree.add([loc.x, loc.y], loc)?;
    }

    // -------- Create empirical semivariogram from a sample
    eprintln!("Empirical semivariogram ...");
    let samples = locs.iter().choose_multiple(&mut rng, args.samples);
    let variogram_bins = empirical_semivariogram(samples, args.nbins, args.range);
    let model = fit_model(variogram_bins);

    // -------- Prediction
    // TODO estimate these
    let cellsize: f64 = 0.0005;
    let extent = (-125.0, 41.5, -124.7, 42.0);

    // Create a 2D raster grid
    let ulorigin = (extent.0, extent.3);
    let llorigin = (extent.0, extent.1);
    let rows = ((extent.3 - extent.1) / cellsize).ceil() as usize;
    let cols = ((extent.2 - extent.0) / cellsize).ceil() as usize;
    eprintln!("Making predictions on grid... {} x {}", rows, cols);

    println!("ncols {}", cols);
    println!("nrows {}", rows);
    // TODO alignment; add halfcell?
    println!("xllcorner {}", llorigin.0);
    println!("yllcorner {}", llorigin.1);
    println!("cellsize {}", cellsize);
    println!("NODATA_value -9999");

    for i in 0..rows {
        let y = ulorigin.1 - (i as f64 * cellsize);
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            let x = ulorigin.0 + (j as f64 * cellsize);
            let pt = (x, y);

            // Identify nearby data points
            let neighbors =
                kdtree.nearest(&[pt.0, pt.1], args.max_neighbors, &squared_euclidean)?;

            // Performing kriging
            let a = create_matrix_a(&neighbors, &model);
            let b = create_vector_b(pt, &neighbors, &model);
            let a_inv = a.try_inverse().unwrap();
            let (predicted_value, _estimation_variance) = predict(&a_inv, &b, &neighbors);
            row.push(predicted_value);

            // GeoJSON
            // println!(
            //     r#"{{"type":"Feature","geometry":{{"type":"Point","coordinates":[{},{}]}},"properties":{{"value":{},"stdev":{}}}}}"#,
            //     pt.0,
            //     pt.1,
            //     predicted_value,
            //     estimation_variance.sqrt()
            // );
        }
        println!("{}", row.iter().join(" "));
    }

    Ok(())
}
