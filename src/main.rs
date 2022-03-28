use std::error::Error;
use std::fs::File;

use clap::Parser;
use itertools::Itertools;
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

    /// Number of locations to sample
    #[clap(short, long, default_value_t = 5000)]
    samples: usize,

    /// Maxiumum Range or Lag distance to consider
    #[clap(short, long, default_value_t = 0.4)]
    range: f64,

    /// Number of lag bins
    #[clap(short, long, default_value_t = 100)]
    nbins: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // Parse XYZ Locations from input text
    let file = File::open(args.points)?;
    let locs = parse_locations(file)?;

    // Random sample
    let mut rng = thread_rng();
    let samples = locs.iter().choose_multiple(&mut rng, args.samples);

    // -------- Create empirical semivariogram
    eprintln!("creating empirical semivariogram...");
    // Create lag bin map, holding the sum and count (partials required to calculate the mean)
    let mut variogram_bins = Vec::with_capacity(args.nbins);
    for b in (0..args.nbins).into_iter() {
        let prop = ((b as f64) + 1.0) / args.nbins as f64;
        let h = args.range * prop;
        variogram_bins.insert(b, LagBin::new(h));
    }

    // Pass 1:
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

        let lagbin_idx = (args.nbins as f64 * (dist / args.range)) as usize; // should always be 0 to args.nbins
        let lagbin = &mut variogram_bins[lagbin_idx];
        lagbin.push(variance);
    }

    // -------- Model specification
    // Create a model fit to the empirical variogram
    // TODO optimize this automatically
    // let model = SphericalVariogramModel::new(0.0, 16500.0, 0.4);
    let model = SphericalVariogramModel::new(2.5, 7.5, 10.0);

    // -------- Prediction (example)
    // right near an observation with a z of -749
    // let pt = (-124.83669, 41.9079);
    let pt = (5.0, 5.0);

    // Identify pairs of data points
    // Rule of thumb: 64 nearest points
    // TODO Note that these pairs do not need to be the same as what created the empirical variogram
    // IOW now that we have a variogram model, we can use a much smaller number of
    // samples in a local window.
    // let samples = locs.iter().choose_multiple(&mut rng, 5);

    // create matrix A
    eprintln!("Create and invert matrix A...");
    let mut matrix_values = Vec::with_capacity((samples.len() + 1).pow(2));
    for si in samples.iter() {
        for sj in samples.iter() {
            // is it more efficient to store the distance matrix in the first pass? defer
            let dist = distance(&si, &sj);
            let modeled_semivariance = model.estimate(dist);
            matrix_values.push(modeled_semivariance);
        }
        // Add a column, ensures the weights sum to one
        matrix_values.push(1.0);
    }
    // Add a final row to ensure weights sum to one
    for _ in (0..samples.len()).into_iter() {
        matrix_values.push(1.0);
    }
    matrix_values.push(0.0);

    // instatiate matrix and take inverse
    let a = DMatrix::from_vec(samples.len() + 1, samples.len() + 1, matrix_values);
    eprintln!("{}", a);
    let a_inv = a.try_inverse().expect("inverting matrix failed");
    eprintln!("{}", a_inv);

    eprintln!("Prediction; create vector b ...");

    // distances from unknown pt to observations
    // plug in distances to the variogram estimator to create vector b
    let mut vector_values = Vec::with_capacity(samples.len() + 1);
    let unknown = Location {
        x: pt.0,
        y: pt.1,
        z: 0.0,
    };
    for s in samples.iter() {
        let dist = distance(&s, &unknown);
        dbg!(dist);
        let modeled_semivariance = model.estimate(dist);
        vector_values.push(modeled_semivariance);
    }
    // Add a final row to ensure weights sum to one
    vector_values.push(1.0);

    let b = DVector::from_vec(vector_values);
    let b2 = b.clone();

    // Do the matrix multiplication
    let weights_vector = a_inv * b;

    // predict the unknown value at pt using sum(weight * observed_z)
    let mut predicted_value = 0.0;
    for (i, s) in samples.iter().enumerate() {
        predicted_value += s.z * weights_vector[i];
    }

    // estimate the kriging error (stddev or sqrt of variance) at pt using
    // Should be possible to do in matrix methods: let estvar = (b2 * weights_vector).sum();
    let mut estimation_variance = 0.0;
    for (i, bsv) in b2.iter().enumerate() {
        estimation_variance += bsv * weights_vector[i];
    }
    dbg!(b2);
    dbg!(weights_vector);

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
    }

    // for lagbin in variogram_bins.iter() {
    //     // Compare empirical vs modelled semivariances
    //     println!("{} {}", lagbin.mean() / 2.0, model.estimate(lagbin.h));
    // }

    // TODO calculate optimal raster grid (affine, shape) based on convex hull and range and cell size (> nugget)
    // TODO interpolate over the grid in parallel, returning 2D array + affine (for rasterio to reconstruct a raster)

    // Extras (futch)
    // geodesic distance, great circle distance, ECEF 3D distance
    // interpolate over hex grid or user-supplied points, returning a 1D array per feature
    // Trend elimination (UK)
    // Subsetting
    // find X observations nearest the unknown
    // Bayesian estimation

    Ok(())
}
