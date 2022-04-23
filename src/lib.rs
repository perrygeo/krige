use std::error::Error;
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

    pub fn estimate(&self, lag: f64) -> f64 {
        let h;
        if lag < 0.0 {
            h = lag.abs();
        } else {
            h = lag;
        }

        if h == 0.0 {
            self.c0 // or zero?
        } else if h > self.a {
            self.c0 + self.c1
        } else {
            self.c0 + self.c1 * (((3.0 * h) / (2.0 * self.a)) - ((h / self.a).powf(3.0) / 2.0))
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

pub fn create_matrix_a(
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

pub fn create_vector_b(
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
    // Should be possible to do in matrix methods: let estvar = (b2 * weights_vector).sum();
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

pub fn render_variogram_text(variogram_bins: &Vec<LagBin>) -> String {
    let pairs: Vec<(f64, f64)> = variogram_bins
        .iter()
        .map(|b| (b.h, b.sum / b.count as f64))
        .collect();

    let s1 = Plot::new(pairs).point_style(PointStyle::new().marker(PointMarker::Cross));
    let v = ContinuousView::new()
        .add(s1)
        // .x_range(-5., 10.)
        // .y_range(-2., 6.)
        .x_label("Lag Distance (h)")
        .y_label("Semivariance");

    Page::single(&v).dimensions(80, 30).to_text().unwrap()
}

pub fn fit_model(
    variogram_bins: Vec<LagBin>,
    init_c1: f64,
    init_a: f64,
) -> SphericalVariogramModel {
    // c1 = sill
    // a = range
    // TODO use empirical data to optimize the parameters
    eprintln!("{}", render_variogram_text(&variogram_bins));
    let init_variogram = SphericalVariogramModel::new(0., init_c1, init_a);
    eprintln!("{:?}", init_variogram);
    init_variogram
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
