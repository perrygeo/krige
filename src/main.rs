// TODO
// fit model automaticaly
//   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy-optimize-curve-fit
// support other models: Exponential, Linear, Gaussian
//   https://mmaelicke.github.io/scikit-gstat/reference/models.html
// parallelize the slow parts
// cross-validation, MSPE on a subset of the original (removed prior to constructing empirical semivariogram)
//
// packaging:
// python interface
// make an API and write unit tests
// error handling
//
// outputs:
// calculate optimal raster grid (affine, shape) based on convex hull and range and cell size (> nugget)
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

use std::error::Error;
use std::fs::File;

use clap::Parser;
use itertools::Itertools;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use nalgebra::{DMatrix, DVector};
use rand::{seq::IteratorRandom, thread_rng};

use rust_interpolate::{
    calculate_variance, distance, parse_locations, LagBin, Location, SphericalVariogramModel,
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
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // Parse XYZ Locations from input text
    eprintln!("Parse XYZ file...");
    let file = File::open(args.points)?;
    let locs = parse_locations(file)?;

    // Random sample
    let mut rng = thread_rng();
    let samples = locs.iter().choose_multiple(&mut rng, args.samples);

    // -------- Create empirical semivariogram
    eprintln!("Empirical semivariogram...");

    // Create lag bin map, holding the sum and count (partials required to calculate the mean)
    let mut variogram_bins = Vec::with_capacity(args.nbins);
    for b in (0..args.nbins).into_iter() {
        let prop = ((b as f64) + 1.0) / args.nbins as f64;
        let h = args.range * prop;
        variogram_bins.insert(b, LagBin::new(h));
    }

    // subset Pass 1:
    // Test only unique pairs within the max range
    // and create the semivariogram
    for pair in samples.iter().combinations(2) {
        let si = pair[0];
        let sj = pair[1];

        let dist = distance(&si, &sj);
        if dist == 0.0 || dist > args.range {
            continue;
        }

        let variance = calculate_variance(&si, &sj);

        // should always be 0 to args.nbins
        let lagbin_idx = (args.nbins as f64 * (dist / args.range)) as usize;

        let lagbin = &mut variogram_bins[lagbin_idx];
        lagbin.push(variance);
    }

    // -------- Create spatial index
    eprint!("Creating spatial index...");
    let mut kdtree = KdTree::new(2);
    // Complete dataset pass 1
    for loc in locs.iter() {
        kdtree.add([loc.x, loc.y], loc)?;
    }
    eprintln!("built kdtree with {} pts", kdtree.size());

    // -------- Model specification
    // Create a model fit to the empirical variogram
    // TODO optimize this automatically
    let model = SphericalVariogramModel::new(0.0, 16500.0, 0.4);

    // -------- Prediction (example)
    // right near an observation with a z of -749
    let pt = (-124.83669, 41.9079);

    // Example Box 6.2 in Burroughs and McDonnel, test.xyz
    // let model = SphericalVariogramModel::new(2.5, 7.5, 10.0);
    // let pt = (5.0, 5.0);

    // Identify nearby data points
    let neighbors = kdtree.nearest(&[pt.0, pt.1], 32, &squared_euclidean)?;

    // create matrix A
    eprintln!("Create and invert matrix A...");
    let num_neighbors = neighbors.len();
    let mut matrix_values = Vec::with_capacity((num_neighbors + 1).pow(2));
    for (_, si) in neighbors.iter() {
        for (_, sj) in neighbors.iter() {
            let dist = distance(&si, &sj);
            let modeled_semivariance = model.estimate(dist);
            matrix_values.push(modeled_semivariance);
        }
        // Add a column, ensures the weights sum to one
        matrix_values.push(1.0);
    }
    // Add a final row to ensure weights sum to one
    for _ in (0..num_neighbors).into_iter() {
        matrix_values.push(1.0);
    }
    matrix_values.push(0.0);

    // instatiate matrix and take inverse
    let a = DMatrix::from_vec(num_neighbors + 1, num_neighbors + 1, matrix_values);
    let a_inv = a.try_inverse().expect("inverting matrix failed");

    // distances from unknown pt to observations
    // plug in distances to the variogram estimator to create vector b
    eprintln!("Prediction; create vector b ...");
    let mut vector_values = Vec::with_capacity(num_neighbors + 1);
    let unknown = Location {
        x: pt.0,
        y: pt.1,
        z: 0.0,
    };
    for (_, s) in neighbors.iter() {
        let dist = distance(&s, &unknown);
        let modeled_semivariance = model.estimate(dist);
        vector_values.push(modeled_semivariance);
    }
    // Add a final row to ensure weights sum to one
    vector_values.push(1.0);

    let b = DVector::from_vec(vector_values);

    // Do the matrix multiplication
    let weights_vector = a_inv * &b;

    // predict the unknown value at pt using sum(weight * observed_z)
    let mut predicted_value = 0.0;
    for (i, (_, s)) in neighbors.iter().enumerate() {
        predicted_value += s.z * weights_vector[i];
    }

    // estimate the kriging variance at a point
    // error in z units is the stddev (the sqrt of variance)
    // Should be possible to do in matrix methods: let estvar = (b2 * weights_vector).sum();
    let mut estimation_variance = 0.0;
    for (i, bsv) in b.iter().enumerate() {
        estimation_variance += bsv * weights_vector[i];
    }

    eprintln!(
        "predicted_value at {:?} is {} with {} standard deviation",
        pt,
        predicted_value,
        estimation_variance.sqrt()
    );

    // -------- Outputs
    for lagbin in variogram_bins.iter() {
        // Empirical Semivariogram
        println!("{} {}", lagbin.h, lagbin.mean() / 2.0,);
        // // Compare empirical vs modelled semivariances
        // println!("{} {}", lagbin.mean() / 2.0, model.estimate(lagbin.h));
    }

    Ok(())
}
