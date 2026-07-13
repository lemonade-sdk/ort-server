cmake_minimum_required(VERSION 3.24)
project(ort-server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pin deliberately; ORT is fetched as a prebuilt archive per platform (CPU EP for v1).
set(ONNXRUNTIME_VERSION "1.27.0" CACHE STRING "ONNX Runtime version")

include(FetchContent)

# --- Header-only deps -------------------------------------------------------
FetchContent_Declare(httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.18.3)
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3)
FetchContent_MakeAvailable(httplib json)

# --- ONNX Runtime (prebuilt, CPU) ------------------------------------------
# Microsoft publishes per-platform CPU archives on the onnxruntime releases page.
if(WIN32)
  set(_ort_pkg "onnxruntime-win-x64-${ONNXRUNTIME_VERSION}")
  set(_ort_ext "zip")
elseif(APPLE)
  set(_ort_pkg "onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}")
  set(_ort_ext "tgz")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(_ort_pkg "onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}")
  set(_ort_ext "tgz")
else()
  set(_ort_pkg "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
  set(_ort_ext "tgz")
endif()
FetchContent_Declare(onnxruntime
  URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${_ort_pkg}.${_ort_ext}")
FetchContent_MakeAvailable(onnxruntime)
set(ORT_ROOT "${onnxruntime_SOURCE_DIR}")

# --- tokenizers-cpp (runtime HF tokenizer from tokenizer.json) -------------
# Loads the model's own tokenizer.json (the exact HuggingFace tokenizer), so any
# HF text classifier runs from a STANDARD ONNX export + its tokenizer.json — no
# per-model in-graph baking, guaranteed tokenization parity. Builds a small Rust
# static lib via cargo (present on CI runners and any dev with rustup); this is a
# build-time dependency only — users of the prebuilt binary never see Rust.
if(WIN32)
  set(TOKENIZERS_CPP_MSVC_RUNTIME_LIBRARY "MD" CACHE STRING "" FORCE)
endif()
FetchContent_Declare(tokenizers_cpp
  GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
  GIT_TAG main)
FetchContent_MakeAvailable(tokenizers_cpp)

# --- Target -----------------------------------------------------------------
add_executable(ort-server src/main.cpp)
target_include_directories(ort-server PRIVATE
  "${ORT_ROOT}/include"
  "${tokenizers_cpp_SOURCE_DIR}/include")
target_link_libraries(ort-server PRIVATE
  httplib::httplib
  nlohmann_json::nlohmann_json
  tokenizers_cpp)

if(WIN32)
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
