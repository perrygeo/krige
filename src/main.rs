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

use std::error::Error;
use std::fs::File;

use clap::Parser;
use itertools::Itertools;
use kdtree::{distance::squared_euclidean, KdTree};
use nalgebra::{DMatrix, DVector};
use rand::{seq::IteratorRandom, thread_rng, Rng};

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

    /// Max Number of neighboring data points to consider
    #[clap(short, long, default_value_t = 64)]
    max_neighbors: usize,
}

fn create_matrix_a(
    neighbors: &Vec<(f64, &&Location)>,
    model: &SphericalVariogramModel,
) -> DMatrix<f64> {
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

    DMatrix::from_vec(num_neighbors + 1, num_neighbors + 1, matrix_values)
}

fn create_vector_b(
    pt: (f64, f64),
    neighbors: &Vec<(f64, &&Location)>,
    model: &SphericalVariogramModel,
) -> DVector<f64> {
    let num_neighbors = neighbors.len();
    let mut vector_values = Vec::with_capacity(num_neighbors + 1);
    let unknown = Location {
        x: pt.0,
        y: pt.1,
        z: 0.0,
    };
    for (_, s) in neighbors.iter() {
        let dist = distance(&s, &unknown);
        // plug in distances to the variogram estimator to create vector b
        let modeled_semivariance = model.estimate(dist);
        vector_values.push(modeled_semivariance);
    }
    // Add a final row to ensure weights sum to one
    vector_values.push(1.0);

    DVector::from_vec(vector_values)
}

fn predict(
    a_inv: &DMatrix<f64>,
    b: &DVector<f64>,
    neighbors: &Vec<(f64, &&Location)>,
) -> (f64, f64) {
    // Do the matrix multiplication
    let weights_vector = a_inv * b;

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

    (predicted_value, estimation_variance)
}

fn empirical_semivariogram(samples: Vec<&Location>, nbins: usize, range: f64) -> Vec<LagBin> {
    // Create lag bin map, holding the sum and count (partials required to calculate the mean)
    let mut variogram_bins = Vec::with_capacity(nbins);
    for b in (0..nbins).into_iter() {
        let prop = ((b as f64) + 1.0) / nbins as f64;
        let h = range * prop;
        variogram_bins.insert(b, LagBin::new(h));
    }

    // subset Pass 1:
    // Test only unique pairs within the max range
    // and create the semivariogram
    for pair in samples.iter().combinations(2) {
        let si = pair[0];
        let sj = pair[1];

        let dist = distance(&si, &sj);
        if dist == 0.0 || dist > range {
            continue;
        }

        let variance = calculate_variance(&si, &sj);

        // should always be 0 to args.nbins
        let lagbin_idx = (nbins as f64 * (dist / range)) as usize;

        let lagbin = &mut variogram_bins[lagbin_idx];
        lagbin.push(variance);
    }
    variogram_bins
}

fn fit_model(_variogram_bins: Vec<LagBin>) -> SphericalVariogramModel {
    // TODO use empirical data to optimize the parameters
    SphericalVariogramModel::new(0., 16500., 0.4)
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
    let mut kdtree = KdTree::new(2);

    const EPSILON: f64 = 1e-9 * 1e-9;

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

    // -------- Prediction (example)
    eprintln!("Making predictions ...");
    let n_predictions = 65536; // 256x256 grid
    for _ in 0..n_predictions {
        let x: f64 = thread_rng().gen_range(-125.02..-124.6);
        let y: f64 = thread_rng().gen_range(41.48..42.02);
        let pt = (x, y);

        // Identify nearby data points
        let neighbors = kdtree.nearest(&[pt.0, pt.1], args.max_neighbors, &squared_euclidean)?;

        // Performing kriging
        let a = create_matrix_a(&neighbors, &model);
        let b = create_vector_b(pt, &neighbors, &model);

        if let Some(a_inv) = a.try_inverse() {
            let (predicted_value, estimation_variance) = predict(&a_inv, &b, &neighbors);

            println!(
                r#"{{"type": "Feature", "geometry": {{ "type": "Point", "coordinates": [{}, {}] }}, "properties": {{ "value": {}, "stdev": {}}}}}"#,
                pt.0,
                pt.1,
                predicted_value,
                estimation_variance.sqrt()
            );
        } else {
            eprintln!("A is NOT INVERTABLE, skipping... ");
        }
    }

    Ok(())
}
