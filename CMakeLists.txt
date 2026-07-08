cmake_minimum_required(VERSION 3.24)
project(ort-server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pin these; bump deliberately. ORT is fetched as a prebuilt archive per platform
# (CPU EP only for v1); onnxruntime-extensions provides the in-graph tokenizer ops.
set(ONNXRUNTIME_VERSION "1.27.0" CACHE STRING "ONNX Runtime version")
set(ORT_EXTENSIONS_TAG   "v0.14.0" CACHE STRING "onnxruntime-extensions git tag")

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

# --- onnxruntime-extensions (in-graph tokenizers) --------------------------
# Built from source; provides the statically-linkable custom-op registration.
#
# SCOPE (deliberate, not a size trim — bundles are only ~4-18 MB): ort-server
# serves TEXT-modality ONNX graphs (classification now; embeddings/reranking
# next). We build the FULL text op set — every tokenizer plus string/regex ops.
# Vision/audio/math ops are excluded by scope: those modalities belong to other
# Lemonade backends (images -> stable-diffusion.cpp; audio -> whisper / moonshine
# / kokoro) and pull heavy deps (OpenCV; DLIB + an FFT backend). Revisit only if
# ort-server ever needs to run a non-text graph.
set(OCOS_ENABLE_CTEST OFF CACHE BOOL "" FORCE)
# The tokenizer C API (OrtxCreateTokenizer/OrtxTokenize) used by src/main.cpp:
set(OCOS_ENABLE_C_API ON CACHE BOOL "" FORCE)
# In scope — text ops:
set(OCOS_ENABLE_SPM_TOKENIZER ON CACHE BOOL "" FORCE)
set(OCOS_ENABLE_BERT_TOKENIZER ON CACHE BOOL "" FORCE)
set(OCOS_ENABLE_WORDPIECE_TOKENIZER ON CACHE BOOL "" FORCE)
set(OCOS_ENABLE_GPT2_TOKENIZER ON CACHE BOOL "" FORCE)
set(OCOS_ENABLE_RE2_REGEX ON CACHE BOOL "" FORCE)
# Out of scope — other backends' modalities (also avoids OpenCV + DLIB/FFT):
set(OCOS_ENABLE_VISION OFF CACHE BOOL "" FORCE)
set(OCOS_ENABLE_CV2 OFF CACHE BOOL "" FORCE)
set(OCOS_ENABLE_OPENCV_CODECS OFF CACHE BOOL "" FORCE)
set(OCOS_ENABLE_AUDIO OFF CACHE BOOL "" FORCE)
set(OCOS_ENABLE_MATH OFF CACHE BOOL "" FORCE)
set(OCOS_ENABLE_DLIB OFF CACHE BOOL "" FORCE)
set(OCOS_ENABLE_BLINGFIRE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(ortextensions
  GIT_REPOSITORY https://github.com/microsoft/onnxruntime-extensions.git
  GIT_TAG ${ORT_EXTENSIONS_TAG})
FetchContent_MakeAvailable(ortextensions)

# --- Target -----------------------------------------------------------------
add_executable(ort-server src/main.cpp)
target_include_directories(ort-server PRIVATE
  "${ORT_ROOT}/include"
  "${ortextensions_SOURCE_DIR}/include")
target_link_libraries(ort-server PRIVATE
  httplib::httplib
  nlohmann_json::nlohmann_json
  onnxruntime_extensions)

if(WIN32)
  target_link_libraries(ort-server PRIVATE "${ORT_ROOT}/lib/onnxruntime.lib")
  add_custom_command(TARGET ort-server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ORT_ROOT}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:ort-server>")
else()
  file(GLOB _ort_libs "${ORT_ROOT}/lib/libonnxruntime.*")
  target_link_libraries(ort-server PRIVATE ${_ort_libs})
endif()
