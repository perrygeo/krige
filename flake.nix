{
  description = "rust-interpolate: ordinary kriging with GDAL I/O";

  inputs = {
    # Pinned nixpkgs (glibc 2.44, GDAL 3.13.3). The gdal crate is pinned to a
    # matching georust/gdal commit (see Cargo.toml).
    nixpkgs.url = "github:NixOS/nixpkgs/58deea654ec5dd0644f31c840a07c29fa0592528";
    flake-utils.url = "github:numtide/flake-utils/b7a79d3fea1e214f6d668d521cecec0c01e26544";
    crane.url = "github:ipetkov/crane/e675a36b7377bebe41d548388a2501cbb16d380d";
  };

  outputs = { self, nixpkgs, flake-utils, crane }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let
        pkgs = import nixpkgs { inherit system; };
        craneLib = crane.mkLib pkgs;

        # Only keep Cargo-relevant files so dependency caching is stable
        # and `.git`/`target` don't leak into the build hash.
        src = craneLib.cleanCargoSource (craneLib.path ./.);

        nativeBuildInputs = with pkgs; [
          cargo
          rustc
          pkg-config
        ];

        buildInputs = with pkgs; [
          gdal
        ];

        commonArgs = {
          inherit src nativeBuildInputs buildInputs;
          strictDeps = true;
        };

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        rust-interpolate = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
        });

        cargoTest = craneLib.cargoTest (commonArgs // {
          inherit cargoArtifacts;
        });
      in
      {
        packages.default = rust-interpolate;
        packages.rust-interpolate = rust-interpolate;

        apps.default = flake-utils.lib.mkApp {
          drv = rust-interpolate;
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ rust-interpolate ];
          nativeBuildInputs = with pkgs; [
            cargo
            rustc
            rustfmt
            clippy
            pkg-config
            # GDAL CLI tools (gdalinfo, ogrinfo, ...) for inspection.
            gdal
          ];
          shellHook = ''
            echo "rust-interpolate dev shell (gdal ${pkgs.gdal.version})"
          '';
        };

        checks.default = cargoTest;
      });
}
