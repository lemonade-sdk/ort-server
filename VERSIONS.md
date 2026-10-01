# Bundled versions

`ort-server` has its own semver. The libraries it bundles are internal build
details recorded here (`GET /health` reports the bundled ONNX Runtime version),
not encoded in the `ort-server` version. See
[docs/UPGRADING-ORT.md](docs/UPGRADING-ORT.md).

| ort-server | ONNX Runtime | Tokenizer |
|------------|--------------|-----------|
| unreleased | 1.27.0       | `third_party/tok_ffi` → HF `tokenizers` 0.22 |
| 0.3.7      | 1.27.0       | tokenizers-cpp c586c52 (pinned) |
| 0.3.6      | 1.27.0       | tokenizers-cpp c586c52 (pinned) |
| 0.3.5      | 1.27.0       | tokenizers-cpp c586c52 (pinned) |
| 0.3.4      | 1.27.0       | tokenizers-cpp c586c52 (pinned) — **not self-contained on Windows (OpenSSL DLLs)** |
| 0.3.3      | 1.27.0       | tokenizers-cpp c586c52 (pinned) |
| 0.3.1-0.3.2 | 1.27.0      | tokenizers-cpp c586c52 (pinned) — **do not use: padded tokenizers served wrong scores** |
| 0.3.0      | 1.27.0       | tokenizers-cpp main (unpinned) |
| 0.2.x      | 1.27.0       | tokenizers-cpp main (unpinned) |

Tokenization loads the model's own `tokenizer.json` at runtime through
`third_party/tok_ffi`, a C ABI this repo owns over the
[HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers) crate —
the same Rust tokenizer `transformers` uses. Exact crate versions are
pinned in `third_party/tok_ffi/Cargo.lock`. The model graph is a plain ONNX
export with no custom operators.
