# Third-party notices

ort-server binaries incorporate or bundle the following third-party software.

| Component | License | Distribution |
|-----------|---------|--------------|
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) © Microsoft Corporation | MIT | Shared library bundled in the release archive |
| [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) © tokenizers-cpp contributors | Apache-2.0 | Statically linked |
| [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) © HuggingFace | Apache-2.0 | Statically linked (via tokenizers-cpp) |
| [SentencePiece](https://github.com/google/sentencepiece) © Google | Apache-2.0 | Statically linked (via tokenizers-cpp) |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) © Yuji Hirose | MIT | Header-only, compiled in |
| [nlohmann/json](https://github.com/nlohmann/json) © Niels Lohmann | MIT | Header-only, compiled in |

Full license texts are available at each project's repository. ONNX Runtime's
license and notices also ship inside the upstream prebuilt package this project
consumes (see `docs/UPGRADING-ORT.md` for the exact version).
