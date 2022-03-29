```
cargo build --release && time ./target/release/rust-interpolate --points ~/work/mosaic1/points.xyz --samples 5800 --range 0.2 --nbins 1200 > /dev/null


cargo build --release && \
  time ./target/release/rust-interpolate --points test.xyz --samples 5 --range 10 --nbins 100 > /tmp/pairs.txt && \
	gnuplot -p -e "plot '/tmp/pairs.txt' with points;"
```
