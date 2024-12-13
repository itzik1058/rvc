{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  outputs =
    { nixpkgs, ... }:
    let
      supportedSystems = [
        "x86_64-linux"
        "x86_64-darwin"
        "aarch64-linux"
        "aarch64-darwin"
      ];
    in
    {
      devShells = nixpkgs.lib.genAttrs supportedSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell rec {
            buildInputs = with pkgs; [
              ffmpeg_6
            ];
            shellHook = with pkgs; ''
              export LD_LIBRARY_PATH=/run/opengl-driver/lib:${lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH
              uv sync
              source .venv/bin/activate
            '';
          };
        }
      );
    };
}
