# Bundled versions

`ort-server` has its own semver. The libraries it bundles are internal build
details recorded here (`GET /health` reports the bundled ONNX Runtime version),
not encoded in the `ort-server` version. See
[docs/UPGRADING-ORT.md](docs/UPGRADING-ORT.md).

| ort-server | ONNX Runtime | Tokenizer |
|------------|--------------|-----------|
| 0.4.0      | 1.27.0       | `third_party/tok_ffi` → HF `tokenizers` 0.22 |
| 0.3.7      | 1.27.0       | tokenizers-cpp c586c52 (pinned) |
| 0.3.6      | 1.27.0       | tokenizers-cpp c586c52 (pinned) |
| 0.3.5      | 1.27.0       | tokenizers-cpp c586c52 (pinned) |
| 0.3.4      | 1.27.0       | tokenizers-cpp c586c52 (pinned) — **not self-contained on Windows (OpenSSL DLLs)** |
| 0.3.3      | 1.27.0       | tokenizers-cpp c586c52 (pinned) |
| 0.3.1-0.3.2 | 1.27.0      | tokenizers-cpp c586c52 (pinned) — **do not use: padded tokenizers served wrong scores** |
| 0.3.0      | 1.27.0       | tokenizers-cpp main (unpinned) |
| 0.2.x      | 1.27.0       | tokenizers-cpp main (unpinned) |

## 0.4.0

- **`/classify` contract change.** `labels` is now required for a
  `zero-shot-classification` model and rejected for any other. Both mismatches
  answer `400`.
- **New response field `truncated`.** Present and `true` only when the server
  cut the input to fit the token window; absent otherwise, never `false`. It is
  reported on **every** task. The encoder path keeps the head and drops the end;
  the zero-shot path keeps head and tail with `\n...\n` between them.
- **New task `zero-shot-classification`.** The label set arrives per request.
  Labels draw on the window first and are refused whole if they do not fit, so
  no label is scored from part of itself.
- **New request limits**, each derived from `max_length`, settable in
  `manifest.json` or on the command line, the flag winning: `max_labels` /
  `--max-labels`, `max_label_chars` / `--max-label-chars`, `max_grid_cells` /
  `--max-grid-cells`. All three are startup errors on a fixed-label model.
- **New `400`s on the zero-shot path:** a control character in a label, a
  duplicate label, an empty or whitespace-only label, a label list the window
  cannot hold, and any of the three limits exceeded.
- **New tokenizer stack.** `third_party/tok_ffi` replaces `tokenizers-cpp`: a C
  ABI this repo owns over the same HuggingFace crate. It passes through the
  character offsets the zero-shot task needs. A `truncation` or `padding` block
  in `tokenizer.json` is now ignored, matching `transformers`' defaults.

Tokenization loads the model's own `tokenizer.json` at runtime through
`third_party/tok_ffi`, a C ABI this repo owns over the
[HuggingFace `tokenizers`](https://github.com/huggingface/tokenizers) crate —
the same Rust tokenizer `transformers` uses. Exact crate versions are
pinned in `third_party/tok_ffi/Cargo.lock`. The model graph is a plain ONNX
export with no custom operators.
