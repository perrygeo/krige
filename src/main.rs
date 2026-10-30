use std::error::Error;

use clap::Parser;
use kdtree::{distance::squared_euclidean, KdTree};
use rand::seq::IteratorRandom;
use rayon::prelude::*;

use rust_interpolate::{
    adjust_extent, create_matrix_a, create_vector_b, empirical_semivariogram, estimated_range,
    estimated_sill, fit_spherical_model, predict, read_locations, write_geotiff,
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Path to a GDAL-readable vector dataset of point observations
    points: String,

    /// Attribute field to use for the observed value (z)
    #[clap(short, long, default_value = "z")]
    field: String,

    /// Vector layer name to read (defaults to the first layer)
    #[clap(short, long)]
    layer: Option<String>,

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

    /// Output the raster grid of predicted values. Default <points>.prediction.tif
    #[clap(short, long)]
    prediction_grid: Option<String>,

    /// Output the raster grid of standard deviations. Default <points>.stddev.tif
    #[clap(short, long)]
    stdev_grid: Option<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let mut rng = rand::rng();

    // Parse point locations from the input vector dataset
    eprintln!("Reading point locations...");
    let (locs, srs) = read_locations(&args.points, args.layer.as_deref(), &args.field)?;

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
        if !nns.is_empty() {
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
        None => default_output(&args.points, "prediction"),
    };
    let stdev_grid = match args.stdev_grid {
        Some(a) => a,
        None => default_output(&args.points, "stddev"),
    };

    eprintln!("Outputs:");
    eprintln!("\tprediction grid: {}", prediction_grid);
    eprintln!("\tstdev grid: {}", stdev_grid);

    // -------- Create empirical semivariogram from a sample
    eprintln!("Empirical semivariogram ...");
    let samples = locs.iter().sample(&mut rng, args.samples);
    let variogram_bins = empirical_semivariogram(samples, args.nbins, range);
    let init_sill = estimated_sill(&variogram_bins);

    let model = fit_spherical_model(&variogram_bins, init_sill, range);
    // TODO
    // let model = fit_gaussian_model(&variogram_bins, init_sill, range);

    // -------- Prediction
    // estimate cellsize such that each row is roughly target_width columns wide
    let cellsize: f64 = (extent[2] - extent[0]) / args.target_width as f64;

    // Create a 2D raster grid
    let halfcell = cellsize / 2.0;
    let ulorigin = (extent[0], extent[3]);
    let rows = ((extent[3] - extent[1]) / cellsize).ceil() as usize;
    let cols = ((extent[2] - extent[0]) / cellsize).ceil() as usize;
    let geotransform = [ulorigin.0, cellsize, 0.0, ulorigin.1, 0.0, -cellsize];

    eprintln!(
        "Making predictions on grid... {} x {} with cellsize {}",
        cols, rows, cellsize
    );

    let mut prediction_data = Vec::with_capacity(rows * cols);
    let mut stdev_data = Vec::with_capacity(rows * cols);

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

        let (prediction_row, stdev_row): (Vec<f64>, Vec<f64>) = row.into_iter().unzip();
        prediction_data.extend(prediction_row);
        stdev_data.extend(stdev_row);
    }

    // Write GeoTIFF outputs
    const NODATA: f64 = -9999.0;
    write_geotiff(
        &prediction_grid,
        cols,
        rows,
        geotransform,
        srs.as_ref(),
        NODATA,
        &prediction_data,
    )?;
    write_geotiff(
        &stdev_grid,
        cols,
        rows,
        geotransform,
        srs.as_ref(),
        NODATA,
        &stdev_data,
    )?;

    Ok(())
}

/// Derive a default output path from an input dataset path.
fn default_output(input: &str, suffix: &str) -> String {
    let path = std::path::Path::new(input);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => {
            format!("{}/{}.{}.tif", parent.display(), stem, suffix)
        }
        _ => format!("{}.{}.tif", stem, suffix),
    }
}
