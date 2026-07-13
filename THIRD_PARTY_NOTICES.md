# Third-party notices

ort-server binaries incorporate or bundle the following third-party software.
Verbatim license texts ship in the `licenses/` directory of every release
archive (vendored in this repo under `third_party_licenses/`, except the ONNX
Runtime texts, which are staged from the exact version-pinned upstream package
at build time).

| Component | License | Text in archive | Distribution |
|-----------|---------|-----------------|--------------|
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) © Microsoft Corporation | MIT | `licenses/onnxruntime-LICENSE`, `licenses/onnxruntime-ThirdPartyNotices.txt` | Shared library bundled in the archive |
| [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) © tokenizers-cpp contributors | Apache-2.0 | `licenses/tokenizers-cpp-LICENSE` | Statically linked |
| [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) © HuggingFace | Apache-2.0 | `licenses/huggingface-tokenizers-LICENSE` | Statically linked (via tokenizers-cpp) |
| [rust-onig / onig_sys](https://github.com/rust-onig/rust-onig) © rust-onig contributors | MIT | `licenses/rust-onig-LICENSE` | Statically linked (via HF tokenizers) |
| [Oniguruma](https://github.com/kkos/oniguruma) © K. Kosako | BSD-2-Clause | `licenses/oniguruma-LICENSE` | Statically linked (vendored by onig_sys) |
| [SentencePiece](https://github.com/google/sentencepiece) © Google | Apache-2.0 | `licenses/sentencepiece-LICENSE` | Statically linked (via tokenizers-cpp) |
| [Abseil](https://github.com/abseil/abseil-cpp) © Google | Apache-2.0 | `licenses/abseil-cpp-LICENSE` | Statically linked (vendored by sentencepiece) |
| [protobuf-lite](https://github.com/protocolbuffers/protobuf) © Google | BSD-3-Clause | `licenses/protobuf-lite-LICENSE` | Statically linked (vendored by sentencepiece) |
| [Darts-clone](https://github.com/s-yata/darts-clone) © Susumu Yata | BSD-2-Clause | `licenses/darts_clone-LICENSE` | Statically linked (vendored by sentencepiece) |
| [esaxx](https://github.com/hillbig/esaxx) © Daisuke Okanohara | MIT | `licenses/esaxx-LICENSE` | Statically linked (vendored by sentencepiece) |
| [msgpack-cxx](https://github.com/msgpack/msgpack-c) © msgpack contributors | BSL-1.0 | `licenses/msgpack-LICENSE` | Header-only, compiled in (via tokenizers-cpp) |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) © Yuji Hirose | MIT | `licenses/cpp-httplib-LICENSE` | Header-only, compiled in |
| [nlohmann/json](https://github.com/nlohmann/json) © Niels Lohmann | MIT | `licenses/nlohmann-json-LICENSE` | Header-only, compiled in |

Exact dependency versions are pinned in `CMakeLists.txt` (SHA256 / commit
SHAs) and recorded in `VERSIONS.md`.
