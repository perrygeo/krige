use std::error::Error;
use std::fs::File;

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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
