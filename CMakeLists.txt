cmake_minimum_required(VERSION 3.24)
project(ort-server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pin deliberately; ORT is fetched as a prebuilt archive per platform (CPU EP for v1).
set(ONNXRUNTIME_VERSION "1.27.0" CACHE STRING "ONNX Runtime version")

include(FetchContent)

# --- Header-only deps -------------------------------------------------------
# httplib auto-links OpenSSL/zlib/brotli when it finds them on the BUILD machine,
# which silently makes the release binary depend on DLLs we do not ship — it then
# dies with STATUS_DLL_NOT_FOUND on any host without them. ort-server serves plain
# HTTP on localhost and needs none of it. (Enforced by the dependency check in the
# release workflow.)
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG a7bc00e3307fecdb4d67545e93be7b88cfb1e186)  # v0.18.3
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03)  # v3.11.3
FetchContent_MakeAvailable(httplib json)

# --- ONNX Runtime (prebuilt, CPU) ------------------------------------------
# Microsoft publishes per-platform CPU archives on the onnxruntime releases page.
# Per-archive SHA256 pins (recompute when bumping ONNXRUNTIME_VERSION — see
# docs/UPGRADING-ORT.md).
if(WIN32)
  set(_ort_pkg "onnxruntime-win-x64-${ONNXRUNTIME_VERSION}")
  set(_ort_ext "zip")
  set(_ort_sha256 "c5c81710938e68079ff1a192b04897faabe4b43830d48f39f27ecd4e16138bfc")
elseif(APPLE)
  set(_ort_pkg "onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}")
  set(_ort_ext "tgz")
  set(_ort_sha256 "545e81c58152353acb0d1e8bd6ce4b62f830c0961f5b3acfedc790ffd76e477a")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(_ort_pkg "onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}")
  set(_ort_ext "tgz")
  set(_ort_sha256 "3e4d83ac06924a32a07b6d7f91ce6f852876153fc0bbdf931bf517a140bfbe48")
else()
  set(_ort_pkg "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
  set(_ort_ext "tgz")
  set(_ort_sha256 "547e40a48f1fe73e3f812d7c88a948612c23f896b91e4e2ee1e232d7b468246f")
endif()
FetchContent_Declare(onnxruntime
  URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${_ort_pkg}.${_ort_ext}"
  URL_HASH SHA256=${_ort_sha256})
FetchContent_MakeAvailable(onnxruntime)
set(ORT_ROOT "${onnxruntime_SOURCE_DIR}")

# --- tokenizer FFI (runtime HF tokenizer from tokenizer.json) --------------
# Loads the model's own tokenizer.json (the exact HuggingFace tokenizer), so any
# HF text classifier runs from a STANDARD ONNX export + its tokenizer.json — no
# per-model in-graph baking, guaranteed tokenization parity.
#
# third_party/tok_ffi is our C ABI over the HuggingFace `tokenizers` crate; it
# also returns the character offsets the zero-shot router needs. Building it
# needs cargo; users of the prebuilt binary never see Rust.
find_program(CARGO_EXECUTABLE cargo
  HINTS "$ENV{USERPROFILE}/.cargo/bin" "$ENV{HOME}/.cargo/bin")
if(NOT CARGO_EXECUTABLE)
  message(FATAL_ERROR
    "cargo not found. ort-server builds a small Rust static library from "
    "third_party/tok_ffi. Install rustup (https://rustup.rs) and re-run; this "
    "is a build-time requirement only.")
endif()

# Rust and the prebuilt ONNX Runtime both use the dynamic CRT (/MD); match it.
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

set(TOK_FFI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tok_ffi")
# Under the binary dir, so the build never writes to the source tree.
set(TOK_FFI_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/tok_ffi")
set(TOK_FFI_LIB
  "${TOK_FFI_TARGET_DIR}/release/${CMAKE_STATIC_LIBRARY_PREFIX}tok_ffi${CMAKE_STATIC_LIBRARY_SUFFIX}")

# Runs cargo on every build and lets it decide what is stale; an up-to-date
# library is left untouched, so nothing relinks. --locked builds exactly what
# Cargo.lock pins, and fails rather than rewriting it.
add_custom_target(tok_ffi_build
  COMMAND "${CARGO_EXECUTABLE}" build --release --locked
          --manifest-path "${TOK_FFI_DIR}/Cargo.toml"
          --target-dir "${TOK_FFI_TARGET_DIR}"
  BYPRODUCTS "${TOK_FFI_LIB}"
  COMMENT "Building tokenizer FFI (Rust)"
  VERBATIM)

# --- Target -----------------------------------------------------------------
add_executable(ort-server src/main.cpp)
add_dependencies(ort-server tok_ffi_build)
target_compile_definitions(ort-server PRIVATE
  ORT_SERVER_ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION}")
target_include_directories(ort-server PRIVATE
  "${ORT_ROOT}/include"
  "${TOK_FFI_DIR}")
target_link_libraries(ort-server PRIVATE
  httplib::httplib
  nlohmann_json::nlohmann_json
  "${TOK_FFI_LIB}")

if(WIN32)
  # The Rust standard library inside tok_ffi.lib needs these. Regenerate
  # the list with:
  #   cargo rustc --release --manifest-path third_party/tok_ffi/Cargo.toml -- \
  #     --print native-static-libs
  target_link_libraries(ort-server PRIVATE
    kernel32 ntdll userenv ws2_32 dbghelp)
  target_link_libraries(ort-server PRIVATE "${ORT_ROOT}/lib/onnxruntime.lib")
  add_custom_command(TARGET ort-server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ORT_ROOT}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:ort-server>")
else()
  file(GLOB _ort_libs "${ORT_ROOT}/lib/libonnxruntime.*")
  target_link_libraries(ort-server PRIVATE ${_ort_libs})
endif()

# The release archive is flat (binary + libonnxruntime side by side); make the
# packaged binary resolve the bundled library relative to itself. BUILD_RPATH
# appends to the automatic build-tree rpath, so build-dir runs keep working.
if(APPLE)
  set_target_properties(ort-server PROPERTIES BUILD_RPATH "@loader_path")
elseif(UNIX)
  set_target_properties(ort-server PROPERTIES BUILD_RPATH "$ORIGIN")
endif()
