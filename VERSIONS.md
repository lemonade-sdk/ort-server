# Bundled versions

`ort-server` has its own semver. The libraries it bundles are internal build
details recorded here (`GET /health` reports the bundled ONNX Runtime version),
not encoded in the `ort-server` version. See
[docs/UPGRADING-ORT.md](docs/UPGRADING-ORT.md).

| ort-server | ONNX Runtime | tokenizers-cpp |
|------------|--------------|----------------|
| 0.3.1+     | 1.27.0       | c586c52 (pinned) |
| 0.3.0      | 1.27.0       | main (unpinned)  |
| 0.2.x      | 1.27.0       | main (unpinned)  |

Tokenization uses [mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)
(loads the model's own `tokenizer.json` at runtime). The model graph is a plain
ONNX export with no custom operators.
