use std::error::Error;
use std::f64::consts::E;
use std::f64::{INFINITY, NEG_INFINITY};
use std::fs::File;

use itertools::Itertools;
use nalgebra::{DMatrix, DVector};

use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{PointMarker, PointStyle};
use plotlib::view::ContinuousView;

use ndarray::prelude::*;

use optimize::Minimizer;
use optimize::NelderMeadBuilder;

/// XYZ Locations
#[derive(Debug, serde::Deserialize)]
pub struct Location {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Clone, Debug)]
pub struct LagBin {
    pub h: f64,
    sum: f64,
    count: usize,
}

impl LagBin {
    pub fn new(h: f64) -> Self {
        LagBin {
            h,
            sum: 0.0,
            count: 0,
        }
    }

    pub fn push(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
    }

    pub fn mean(&self) -> f64 {
        self.sum / (self.count as f64)
    }
}

pub trait VariogramModel {
    fn estimate(&self, lag: f64) -> f64;
}

/// Spherical Semivariogram Model
/// Appropriate for spatial phenomenon with a clear transition point
#[derive(Clone, Debug)]
pub struct SphericalVariogramModel {
    c0: f64,
    c1: f64,
    a: f64,
}

impl SphericalVariogramModel {
    pub fn new(c0: f64, c1: f64, a: f64) -> Self {
        SphericalVariogramModel { c0, c1, a }
    }
}

impl VariogramModel for SphericalVariogramModel {
    fn estimate(&self, lag: f64) -> f64 {
        let h = lag.abs();

        if h == 0.0 {
            // Lag of zero implies exact same point, use the nugget
            self.c0
        } else if h > self.a {
            // Lag greater than range, use the nugget + sill
            self.c0 + self.c1
        } else {
            // Lag is within range, calcululate
            self.c0 + self.c1 * (((3.0 * h) / (2.0 * self.a)) - ((h / self.a).powf(3.0) / 2.0))
        }
    }
}

/// Gaussian Semivariogram Model
/// Appropriate for spatial phenomenon with a smoothly varying pattern
#[derive(Clone, Debug)]
pub struct GaussianVariogramModel {
    c0: f64,
    c1: f64,
    a: f64,
}

impl GaussianVariogramModel {
    pub fn new(c0: f64, c1: f64, a: f64) -> Self {
        GaussianVariogramModel { c0, c1, a }
    }
}

impl VariogramModel for GaussianVariogramModel {
    fn estimate(&self, lag: f64) -> f64 {
        let h = lag.abs();

        if h == 0.0 {
            // Lag of zero implies exact same point, use the nugget
            self.c0
        } else {
            // Lag is within range, calculate
            self.c0 + self.c1 * (1. - E.powf(-1. * h.powf(2.) / self.a.powf(2.)))
        }
    }
}

pub fn calculate_variance(si: &Location, sj: &Location) -> f64 {
    (si.z - sj.z).powf(2.0)
}

pub fn distance(a: &Location, b: &Location) -> f64 {
    (((a.x - b.x).powf(2.0)) + ((a.y - b.y).powf(2.0))).sqrt()
}

pub fn parse_locations(file: File) -> Result<Vec<Location>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .trim(csv::Trim::All)
        .delimiter(b'\t')
        .from_reader(file);

    let mut locations: Vec<Location> = Vec::new();
    for result in rdr.deserialize() {
        let loc: Location = result?;
        locations.push(loc);
    }

    Ok(locations)
}

pub fn create_matrix_a<V: VariogramModel>(
    neighbors: &Vec<(f64, &&Location)>,
    model: &V,
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

pub fn create_vector_b<V: VariogramModel>(
    pt: (f64, f64),
    neighbors: &Vec<(f64, &&Location)>,
    model: &V,
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

pub fn predict(
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
    // TODO should be possible to do in matrix methods: let estvar = (b2 * weights_vector).sum();
    let mut estimation_variance = 0.0;
    for (i, bsv) in b.iter().enumerate() {
        estimation_variance += bsv * weights_vector[i];
    }

    (predicted_value, estimation_variance)
}

pub fn empirical_semivariogram(samples: Vec<&Location>, nbins: usize, range: f64) -> Vec<LagBin> {
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

pub fn render_variogram_text<V: VariogramModel>(variogram_bins: &Vec<LagBin>, model: &V) -> String {
    let observed: Vec<(f64, f64)> = variogram_bins.iter().map(|b| (b.h, b.mean())).collect();

    let modeled: Vec<(f64, f64)> = variogram_bins
        .iter()
        .map(|b| (b.h, model.estimate(b.h)))
        .collect();

    let s1 = Plot::new(observed).point_style(PointStyle::new().marker(PointMarker::Cross));
    let s2 = Plot::new(modeled).point_style(PointStyle::new().marker(PointMarker::Circle));
    let v = ContinuousView::new()
        .add(s1)
        .add(s2)
        // .x_range(-5., 10.)
        // .y_range(-2., 6.)
        .x_label("Lag Distance (h)")
        .y_label("Semivariance");

    Page::single(&v).dimensions(80, 30).to_text().unwrap()
}

pub fn fit_spherical_model(
    variogram_bins: &Vec<LagBin>,
    init_c1: f64,
    init_a: f64,
) -> SphericalVariogramModel {
    // c0 = nugget
    // c1 = sill
    // a = range

    // Our objective function to minimize
    // Use clojure to capture the empirical variogram_bins
    let objective_function = |params: ArrayView1<f64>| {
        // Create model
        let c0 = params[0];
        let c1 = params[1];
        let a = params[2];
        let variogram = SphericalVariogramModel::new(c0, c1, a);

        // Iterate through variogram bins and measure residual sum of squares
        let rss: Vec<f64> = variogram_bins
            .iter()
            .map(|bin| {
                let actual = bin.mean();
                if actual.is_nan() {
                    0.
                } else {
                    let estimated = variogram.estimate(bin.h);
                    // not true sum of squares - we scale by the lag dist
                    // to avoid the influence of outliers at high h
                    ((actual - estimated) / bin.h).powf(2.0)
                }
            })
            .collect();

        // Root Mean Square Error
        let n = rss.len();
        let rmse = (rss.into_iter().sum::<f64>() / n as f64).sqrt();

        if c0 < 0. {
            // Penalize negatives to enforce the constraint
            rmse * -200. * c0
        } else {
            rmse
        }
    };

    // Set the starting guess
    let args = Array::from_vec(vec![0.0, init_c1, init_a]);

    // Run the optimization
    let minimizer = NelderMeadBuilder::default()
        .maxiter(80_000)
        .maxfun(80_000)
        .ftol(1000.)
        .build()
        .unwrap();
    let params = minimizer.minimize(&objective_function, args.view());

    // Create model
    let c0 = params[0];
    let c0 = {
        if c0 < 0. {
            0.
        } else {
            c0
        }
    };

    let c1 = params[1];
    let a = params[2];
    let model = SphericalVariogramModel::new(c0, c1, a);
    eprintln!("{}", render_variogram_text(&variogram_bins, &model));
    eprintln!("{:?}", model);
    model
}

pub fn fit_gaussian_model(
    variogram_bins: &Vec<LagBin>,
    init_c1: f64,
    init_a: f64,
) -> GaussianVariogramModel {
    // c0 = nugget
    // c1 = sill
    // a = range

    // Our objective function to minimize
    // Use clojure to capture the empirical variogram_bins
    let objective_function = |params: ArrayView1<f64>| {
        // Create model
        let c0 = params[0];
        let c1 = params[1];
        let a = params[2];
        let variogram = GaussianVariogramModel::new(c0, c1, a);

        // Iterate through variogram bins and measure residual sum of squares
        let rss: Vec<f64> = variogram_bins
            .iter()
            .map(|bin| {
                let actual = bin.mean();
                if actual.is_nan() {
                    0.
                } else {
                    let estimated = variogram.estimate(bin.h);
                    // not true sum of squares - we scale by the lag dist
                    // to avoid the influence of outliers at high h
                    ((actual - estimated) / bin.h).powf(2.0)
                }
            })
            .collect();

        // Root Mean Square Error
        let n = rss.len();
        let rmse = (rss.into_iter().sum::<f64>() / n as f64).sqrt();

        if c0 < 0. {
            // Penalize negatives to enforce the constraint
            rmse * -200. * c0
        } else {
            rmse
        }
    };

    // Set the starting guess
    let args = Array::from_vec(vec![0.0, init_c1, init_a]);

    // Run the optimization
    let minimizer = NelderMeadBuilder::default()
        .maxiter(80_000)
        .maxfun(80_000)
        .ftol(1000.)
        .build()
        .unwrap();
    let params = minimizer.minimize(&objective_function, args.view());

    // Create model
    let c0 = params[0];
    let c0 = {
        if c0 < 0. {
            0.
        } else {
            c0
        }
    };

    let c1 = params[1];
    let a = params[2];
    let model = GaussianVariogramModel::new(c0, c1, a);
    eprintln!("{}", render_variogram_text(&variogram_bins, &model));
    eprintln!("{:?}", model);
    model
}

pub fn estimated_sill(bins: &Vec<LagBin>) -> f64 {
    let mut maxval: f64 = NEG_INFINITY;
    let mut minval: f64 = INFINITY;
    for bin in bins.iter() {
        let val = bin.mean();
        if val > maxval {
            maxval = val;
        }
        if val < minval {
            minval = val;
        }
    }
    (maxval + minval) / 2.0
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
