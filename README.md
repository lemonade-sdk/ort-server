# ort-server

A generic, self-contained **ONNX Runtime model server** for [Lemonade](https://github.com/lemonade-sdk/lemonade). It loads one exported ONNX model + a small manifest and serves it over HTTP. Lemonade launches it as a subprocess (like `moonshine-server`) and forwards requests to it.

**Consumer:** lemonade's `onnxruntime` backend (`/v1/classify`, [#2592](https://github.com/lemonade-sdk/lemonade/issues/2592)). The router's `classifier` condition type will consume that endpoint once [#2384](https://github.com/lemonade-sdk/lemonade/issues/2384) wires it. Classification is the first capability; the server is intentionally generic so embeddings and reranking can follow.

> Status: **experimental** — released and consumed by Lemonade's `onnxruntime` backend. See [#2592](https://github.com/lemonade-sdk/lemonade/issues/2592).

## Scope

ort-server serves **text-modality** ONNX graphs — classification today, embeddings
and reranking next. Because it tokenizes from the model's own `tokenizer.json`,
any HuggingFace text classifier runs from a standard ONNX export with no per-model
work. Vision and audio models are **out of scope by design**: those modalities are
served by other Lemonade backends (images → stable-diffusion.cpp; audio →
whisper / moonshine / kokoro).

## Design

The server is thin, and a model is easy to bring: **a stock Optimum export runs
as-is** — `optimum-cli export onnx --model <hf_id> <dir>` and point ort-server
at the directory.

- `model.onnx` is a plain export (`input_ids`/`attention_mask`[/`token_type_ids`] → logits) — the ordinary `optimum` export, no in-graph baking, no custom ops.
- The server tokenizes at runtime by loading the model's `tokenizer.json` via [mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) — the exact HuggingFace tokenizer, so parity is guaranteed.
- The output contract (task, labels, normalization, token budget) is inferred
  from the export's own `config.json` + `tokenizer_config.json`, honoring HF
  `problem_type` semantics (`multi_label_classification` → sigmoid; regression
  heads are rejected — no label scores in [0,1]).
- An optional **`manifest.json`** overrides the inference; when present it is
  the contract and is validated strictly.

```
model-dir/
  model.onnx             # plain export: input_ids/attention_mask -> logits
  tokenizer.json         # the model's HuggingFace tokenizer
  config.json            # stock HF config (id2label / problem_type / architectures)
  tokenizer_config.json  # model_max_length
  manifest.json          # OPTIONAL explicit override of the inferred contract
```

`manifest.json` override (validated at startup — unknown values are a startup
error, and the model's output dimension must match `id2label` at inference time):
```json
{
  "task": "text-classification",        // or "token-classification"
  "id2label": {"0": "SAFE", "1": "INJECTION"},
  "score_normalization": "softmax",     // "softmax" (default) | "sigmoid"
  "token_aggregation": null,            // token-classification: "max" (default) | "mean"
  "max_length": 512                     // optional token budget; longer inputs are truncated
}
```

## HTTP contract

| Method | Path | Body | Response |
|--------|------|------|----------|
| GET | `/health` | — | `200` when the model is loaded and ready |
| POST | `/classify` | `{"text": "...", "top_k": N?}` | `{"labels": {"<label>": <score in [0,1]>, ...}}` — `top_k` omitted or `0` returns all labels |

Future capabilities (same server, new endpoints): `POST /embed`, `POST /rerank`.

## Model catalog tooling

`tools/classifier_catalog/` holds the Python tooling that produces the
lemonade-sdk HF catalog for this server: `export.py` (Optimum export +
manifest + parity validation vs the PyTorch reference), `publish.py`
(fail-closed license allowlist + parity gates + model cards), and the
exploratory `export_and_benchmark.py` harness. It lives in this repo so the
manifest contract's producer and consumer version together.

## Execution providers

v1 ships **CPU EP** only. The roadmap adds providers as build variants without changing the code:

`cpu` → `dml` (Windows GPU/NPU) → `rocm` (AMD GPU) → `vitisai` (Ryzen NPU).

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/ort-server --model-path <model-dir> --port 8100
```

CMake fetches ONNX Runtime, tokenizers-cpp, cpp-httplib, and nlohmann/json (see `CMakeLists.txt`). tokenizers-cpp builds a small Rust static lib, so **cargo/rustup must be on PATH** to build ort-server from source (build-time only — users of the prebuilt binary need nothing).

## Releases

CPU-only, so release bundles build on **GitHub-hosted runners** (`windows-latest`, `ubuntu-latest`, `macos-latest`) — no self-hosted/GPU runners for v1. Pushing a `v*` tag builds and attaches per-platform archives named to match Lemonade's downloader:

```
ort-server-<version>-windows-x64.zip
ort-server-<version>-linux-x64.tar.gz
ort-server-<version>-linux-arm64.tar.gz
ort-server-<version>-macos-arm64.tar.gz
```

## License

[Apache-2.0](LICENSE). Release archives bundle the ONNX Runtime shared library
and statically link the tokenizer stack; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
