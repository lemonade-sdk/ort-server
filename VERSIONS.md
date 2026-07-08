# Bundled versions

`ort-server` has its own semver. The ONNX Runtime and onnxruntime-extensions it
bundles are internal build details recorded here (and reported by `GET /health`),
not encoded in the `ort-server` version. See [docs/UPGRADING-ORT.md](docs/UPGRADING-ORT.md).

| ort-server | ONNX Runtime | onnxruntime-extensions |
|------------|--------------|------------------------|
| 0.1.x      | 1.27.0       | v0.14.0                |
