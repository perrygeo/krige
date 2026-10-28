use std::error::Error;
use std::f64::consts::E;
use std::fs::File;

use itertools::Itertools;
use nalgebra::{DMatrix, DVector};

use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{PointMarker, PointStyle};
use plotlib::view::ContinuousView;

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
            // Lag of zero implies exact same point, use zero
            0.
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
            // Lag of zero implies exact same point, use zero
            0.
        } else {
            // use 4/7 range per
            // https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/variogram_models.html
            self.c0 + self.c1 * (1. - E.powf(-h.powf(2.) / (self.a * 4. / 7.).powf(2.)))
        }
    }
}

pub fn calculate_variance(si: &Location, sj: &Location) -> f64 {
    (si.z - sj.z).powf(2.0)
}

pub fn distance(a: &Location, b: &Location) -> f64 {
    (((a.x - b.x).powf(2.0)) + ((a.y - b.y).powf(2.0))).sqrt()
}

pub fn adjust_extent(extent: &mut [f64], x: f64, y: f64) {
    if x < extent[0] {
        extent[0] = x;
    } else if x > extent[2] {
        extent[2] = x;
    }

    if y < extent[1] {
        extent[1] = y;
    } else if y > extent[3] {
        extent[3] = y;
    }
}

pub fn estimated_range(extent: &[f64; 4]) -> f64 {
    let width = extent[2] - extent[0];
    let height = extent[3] - extent[1];
    (height + width) / 1.5
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
            let dist = distance(si, sj);
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
        let dist = distance(s, &unknown);
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
    let mut estimation_variance = 0.0;

    for (i, (_, s)) in neighbors.iter().enumerate() {
        predicted_value += weights_vector[i] * s.z;
        estimation_variance += weights_vector[i] * b[i];
    }

    // Add theta term
    estimation_variance += weights_vector[neighbors.len()];

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

        let dist = distance(si, sj);
        if dist == 0.0 || dist > range {
            continue;
        }

        let variance = calculate_variance(si, sj);

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

fn order_simplex(sim: &mut [Vec<f64>], fsim: &mut [f64]) {
    let mut order: Vec<usize> = (0..fsim.len()).collect();
    order.sort_by(|&a, &b| {
        fsim[a]
            .partial_cmp(&fsim[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_sim: Vec<Vec<f64>> = order.iter().map(|&i| sim[i].clone()).collect();
    let sorted_fsim: Vec<f64> = order.iter().map(|&i| fsim[i]).collect();
    sim.clone_from_slice(&sorted_sim);
    fsim.clone_from_slice(&sorted_fsim);
}

/// Nelder-Mead simplex minimization (ported from the `optimize` crate to drop
/// its ndarray 0.11 dependency). Minimizes `func` starting from `x0`.
fn nelder_mead<F>(
    func: F,
    x0: &[f64],
    maxiter: usize,
    maxfun: usize,
    xtol: f64,
    ftol: f64,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let rho = 1.0f64;
    let chi = 2.0f64;
    let psi = 0.5f64;
    let sigma = 0.5f64;

    let nonzdelt = 0.05f64;
    let zdelt = 0.00025f64;

    // Build the initial simplex (x0 plus one perturbed point per dimension).
    let mut sim: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    let mut fsim: Vec<f64> = Vec::with_capacity(n + 1);
    let mut num_calls = 0usize;

    sim.push(x0.to_vec());
    fsim.push(func(x0));
    num_calls += 1;
    for k in 0..n {
        let mut y = x0.to_vec();
        y[k] = if y[k] == 0.0 {
            zdelt
        } else {
            (1.0 + nonzdelt) * y[k]
        };
        fsim.push(func(&y));
        num_calls += 1;
        sim.push(y);
    }

    order_simplex(&mut sim, &mut fsim);

    let mut iterations = 1usize;

    loop {
        if maxiter <= iterations || maxfun <= num_calls {
            break;
        }

        // xtol: max absolute deviation from best point in parameter space.
        let mut max_diff = 0.0f64;
        for j in 1..=n {
            for k in 0..n {
                let d = (sim[j][k] - sim[0][k]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        if max_diff < xtol {
            break;
        }

        // ftol: max absolute deviation from best function value.
        let mut max_fdiff = 0.0f64;
        for j in 1..=n {
            let d = (fsim[j] - fsim[0]).abs();
            if d > max_fdiff {
                max_fdiff = d;
            }
        }
        if max_fdiff < ftol {
            break;
        }

        // Centroid of all points except the worst.
        let xbar: Vec<f64> = (0..n)
            .map(|k| sim[..n].iter().map(|p| p[k]).sum::<f64>() / n as f64)
            .collect();

        // Reflection.
        let xr: Vec<f64> = (0..n)
            .map(|k| xbar[k] * (rho + 1.0) - sim[n][k] * rho)
            .collect();
        let fxr = func(&xr);
        num_calls += 1;

        let mut doshrink = false;

        if fxr < fsim[0] {
            // Expansion.
            let xe: Vec<f64> = (0..n)
                .map(|k| (1.0 + rho * chi) * xbar[k] - rho * chi * sim[n][k])
                .collect();
            let fxe = func(&xe);
            num_calls += 1;

            if fxe < fxr {
                sim[n] = xe;
                fsim[n] = fxe;
            } else {
                sim[n] = xr;
                fsim[n] = fxr;
            }
        } else if fxr < fsim[n - 1] {
            sim[n] = xr;
            fsim[n] = fxr;
        } else if fxr < fsim[n] {
            // Outside contraction.
            let xc: Vec<f64> = (0..n)
                .map(|k| (1.0 + psi * rho) * xbar[k] - psi * rho * sim[n][k])
                .collect();
            let fxc = func(&xc);
            num_calls += 1;

            if fxc > fxr {
                doshrink = true;
            } else {
                sim[n] = xc;
                fsim[n] = fxc;
            }
        } else {
            // Inside contraction.
            let xcc: Vec<f64> = (0..n)
                .map(|k| (1.0 - psi) * xbar[k] + psi * sim[n][k])
                .collect();
            let fxcc = func(&xcc);
            num_calls += 1;

            if fxcc < fsim[n] {
                sim[n] = xcc;
                fsim[n] = fxcc;
            } else {
                doshrink = true;
            }
        }

        if doshrink {
            for j in 1..=n {
                let mut tmp: Vec<f64> = (0..n).map(|k| sim[j][k] - sim[0][k]).collect();
                for k in 0..n {
                    tmp[k] = tmp[k] * sigma + sim[0][k];
                }
                sim[j] = tmp.clone();
                fsim[j] = func(&tmp);
                num_calls += 1;
            }
        }

        order_simplex(&mut sim, &mut fsim);
        iterations += 1;
    }

    sim[0].clone()
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
    let objective_function = |params: &[f64]| {
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

    // Run the optimization
    let params = nelder_mead(
        objective_function,
        &[0.0, init_c1, init_a],
        80_000,
        80_000,
        1e-4,
        1000.,
    );

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
    eprintln!("{}", render_variogram_text(variogram_bins, &model));
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
    let objective_function = |params: &[f64]| {
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

    // Run the optimization
    let params = nelder_mead(
        objective_function,
        &[0.0, init_c1, init_a],
        80_000,
        80_000,
        1e-4,
        1000.,
    );

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
    eprintln!("{}", render_variogram_text(variogram_bins, &model));
    eprintln!("{:?}", model);
    model
}

pub fn estimated_sill(bins: &Vec<LagBin>) -> f64 {
    let mut maxval: f64 = f64::NEG_INFINITY;
    let mut minval: f64 = f64::INFINITY;
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
    use crate::GaussianVariogramModel;
    use crate::SphericalVariogramModel;
    use crate::VariogramModel;

    fn approx_equal(a: f64, b: f64) -> bool {
        let decimal_places = 6;
        let factor = 10.0f64.powi(decimal_places);
        let a = (a * factor).trunc();
        let b = (b * factor).trunc();
        a == b
    }

    #[test]
    fn spherical_model_estimation() {
        let m = SphericalVariogramModel::new(0., 10., 100.);
        assert_eq!(m.estimate(1.), m.estimate(-1.));
        assert!(approx_equal(m.estimate(0.), 0.));
        assert!(approx_equal(m.estimate(50.), 6.875));
        assert!(approx_equal(m.estimate(100.), 10.));
        assert!(approx_equal(m.estimate(200.), 10.));
    }

    #[test]
    fn gaussian_model_estimation() {
        let m = GaussianVariogramModel::new(0., 10., 100.);
        assert_eq!(m.estimate(1.), m.estimate(-1.));
        assert!(approx_equal(m.estimate(0.), 0.));
        assert!(approx_equal(m.estimate(50.), 5.34956811));
        assert!(approx_equal(m.estimate(100.), 9.53229377));
        assert!(approx_equal(m.estimate(200.), 9.99995214));
        assert!(approx_equal(m.estimate(800.), 10.));
    }
}
