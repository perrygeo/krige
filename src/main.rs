use itertools::Itertools;
use std::error::Error;
use std::fs::File;
use std::io::Write;

use clap::Parser;
use kdtree::{distance::squared_euclidean, KdTree};
use rand::{seq::IteratorRandom, thread_rng};
use rayon::prelude::*;

use rust_interpolate::{
    adjust_extent, create_matrix_a, create_vector_b, empirical_semivariogram, estimated_range,
    estimated_sill, fit_gaussian_model, fit_spherical_model, parse_locations, predict,
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Filename of the XYZ tab delimited text file
    points: String,

    /// Number of locations to sample for the empirical semivariogram
    #[clap(short, long, default_value_t = 5000)]
    samples: usize,

    /// Number of lag bins for the empirical semivariogram
    #[clap(short, long, default_value_t = 256)]
    nbins: usize,

    /// Max Number of neighboring data points to consider
    #[clap(short, long, default_value_t = 16)]
    max_neighbors: usize,

    /// Target width of the output raster
    #[clap(short, long, default_value_t = 256)]
    target_width: u32,

    // optional, defaults calculated at runtime
    /// Maxiumum Range or Lag distance to consider
    #[clap(short, long)]
    range: Option<f64>,

    /// Output the raster grid of predicted values. Default <points>.prediction.grd
    #[clap(short, long)]
    prediction_grid: Option<String>,

    /// Output the raster grid of standard deviations. Default <points>.stddev.grd
    #[clap(short, long)]
    stdev_grid: Option<String>,
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

    // while we're at it, calculate the bounding coordinates
    let mut extent = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for loc in locs.iter() {
        // If existing points, check to avoid dups, there must be a better way!
        let nns = kdtree.nearest(&[loc.x, loc.y], 1, &squared_euclidean)?;
        if nns.len() > 0 {
            let (sqdist, _existing) = nns[0];
            if sqdist < EPSILON {
                // Duplicate point exists, skip
                continue;
            }
        }
        // New unique point
        kdtree.add([loc.x, loc.y], loc)?;
        adjust_extent(&mut extent, loc.x, loc.y);
    }
    eprintln!("extent {:?}", extent);

    // Calculate defaults if not provided
    let range = match args.range {
        Some(a) => a,
        None => estimated_range(&extent),
    };
    let prediction_grid = match args.prediction_grid {
        Some(a) => a,
        None => String::from("/tmp/predictions.grd"),
    };
    let stdev_grid = match args.stdev_grid {
        Some(a) => a,
        None => String::from("/tmp/stddev.grd"),
    };

    eprintln!("Outputs:");
    eprintln!("\tprediction grid: {}", prediction_grid);
    eprintln!("\tstdev grid: {}", stdev_grid);

    let mut prediction_file = File::create(prediction_grid)?;
    let mut stdev_file = File::create(stdev_grid)?;

    // -------- Create empirical semivariogram from a sample
    eprintln!("Empirical semivariogram ...");
    let samples = locs.iter().choose_multiple(&mut rng, args.samples);
    let variogram_bins = empirical_semivariogram(samples, args.nbins, range);
    let init_sill = estimated_sill(&variogram_bins);

    let model = fit_spherical_model(&variogram_bins, init_sill, range);
    // TODO
    // let model = fit_gaussian_model(&variogram_bins, init_sill, range);

    // -------- Prediction
    // estimate cellsize such that each row is roughly x columns wide to cover the extent
    let cellsize: f64 = (extent[2] - extent[0]) / args.target_width as f64;

    // Create a 2D raster grid
    let halfcell = cellsize / 2.0;
    let ulorigin = (extent[0], extent[3]);
    let rows = ((extent[3] - extent[1]) / cellsize).ceil() as usize;
    let cols = ((extent[2] - extent[0]) / cellsize).ceil() as usize;
    // alignment in QGIS seems to require the upper-left corner of the lower-left cell
    // whereas the ascii grid spec says LOWER-left corner of the lower-left cell
    let llcorner = (ulorigin.0, ulorigin.1 - (cellsize * rows as f64));

    // Write headers
    writeln!(&mut prediction_file, "ncols {}", cols)?;
    writeln!(&mut prediction_file, "nrows {}", rows)?;
    writeln!(&mut prediction_file, "xllcorner {}", llcorner.0)?;
    writeln!(&mut prediction_file, "yllcorner {}", llcorner.1)?;
    writeln!(&mut prediction_file, "cellsize {}", cellsize)?;
    writeln!(&mut prediction_file, "NODATA_value -9999")?;
    writeln!(&mut stdev_file, "ncols {}", cols)?;
    writeln!(&mut stdev_file, "nrows {}", rows)?;
    writeln!(&mut stdev_file, "xllcorner {}", llcorner.0)?;
    writeln!(&mut stdev_file, "yllcorner {}", llcorner.1)?;
    writeln!(&mut stdev_file, "cellsize {}", cellsize)?;
    writeln!(&mut stdev_file, "NODATA_value -9999")?;

    eprintln!(
        "Making predictions on grid... {} x {} with cellsize {}",
        cols, rows, cellsize
    );

    for i in 0..rows {
        let y = ulorigin.1 - (i as f64 * cellsize) - halfcell;
        let row: Vec<(f64, f64)> = (0..cols)
            .into_par_iter()
            .map(|j| {
                let x = ulorigin.0 + (j as f64 * cellsize) + halfcell;
                let pt = (x, y);

                // Identify nearby data points
                let neighbors = kdtree
                    .nearest(&[pt.0, pt.1], args.max_neighbors, &squared_euclidean)
                    .unwrap();

                // Perform kriging
                let a = create_matrix_a(&neighbors, &model);
                let b = create_vector_b(pt, &neighbors, &model);
                let a_inv = a.try_inverse().unwrap();
                let (predicted_value, estimation_variance) = predict(&a_inv, &b, &neighbors);

                (predicted_value, estimation_variance.sqrt())
            })
            .collect();

        // Write to ascii grids
        let (prediction_row, stdev_row): (Vec<_>, Vec<_>) = row.into_iter().unzip();
        writeln!(&mut prediction_file, "{}", prediction_row.iter().join(" "))?;
        writeln!(&mut stdev_file, "{}", stdev_row.iter().join(" "))?;
    }

    Ok(())
}
