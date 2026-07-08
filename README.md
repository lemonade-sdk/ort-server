# ort-server

A generic, self-contained **ONNX Runtime model server** for [Lemonade](https://github.com/lemonade-sdk/lemonade). It loads one exported ONNX model + a small manifest and serves it over HTTP. Lemonade launches it as a subprocess (like `moonshine-server`) and forwards requests to it.

**Consumer:** lemonade [#2592](https://github.com/lemonade-sdk/lemonade/issues/2592) — the router's `classifier` condition type. Classification is the first capability; the server is intentionally generic so embeddings and reranking can follow.

> Status: **initial scaffold** — not yet built/released. See [#2592](https://github.com/lemonade-sdk/lemonade/issues/2592).

## Design

The server is thin. Genericity lives in the **artifacts**, not the code:

- Each model ships a self-contained ONNX graph with **tokenization + pre/post baked in** (via [onnxruntime-extensions](https://github.com/microsoft/onnxruntime-extensions)) — so the graph takes a **raw string** and emits logits. The server never implements a tokenizer.
- A `manifest.json` beside the model declares the task and how to shape the output.

```
model-dir/
  model.onnx        # string-in -> logits (tokenizer baked via ort-extensions)
  manifest.json
```

`manifest.json`:
```json
{
  "task": "text-classification",        // or "token-classification"
  "id2label": {"0": "SAFE", "1": "INJECTION"},
  "score_normalization": "softmax",
  "token_aggregation": null              // "first-subword" | "max" for token-classification
}
```

## HTTP contract

| Method | Path | Body | Response |
|--------|------|------|----------|
| GET | `/health` | — | `200` when the model is loaded and ready |
| POST | `/classify` | `{"text": "...", "top_k": N?}` | `{"labels": {"<label>": <score in [0,1]>, ...}}` |

Future capabilities (same server, new endpoints): `POST /embed`, `POST /rerank`.

## Execution providers

v1 ships **CPU EP** only. The roadmap adds providers as build variants without changing the code:

`cpu` → `dml` (Windows GPU/NPU) → `rocm` (AMD GPU) → `vitisai` (Ryzen NPU).

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/ort-server --model-path <model-dir> --port 8100
```

CMake fetches ONNX Runtime, onnxruntime-extensions, cpp-httplib, and nlohmann/json (see `CMakeLists.txt`). Pin versions there.

## Releases

CPU-only, so release bundles build on **GitHub-hosted runners** (`windows-latest`, `ubuntu-latest`, `macos-latest`) — no self-hosted/GPU runners for v1. Pushing a `v*` tag builds and attaches per-platform archives named to match Lemonade's downloader:

```
ort-server-<version>-windows-x64.zip
ort-server-<version>-linux-x64.tar.gz
ort-server-<version>-linux-arm64.tar.gz
ort-server-<version>-macos-arm64.tar.gz
```

## License

TODO: adopt the lemonade-sdk project license before first public release.
