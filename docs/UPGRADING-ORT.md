# Upgrading the ONNX Runtime version

`ort-server` bundles a specific ONNX Runtime (ORT) and statically links the
tokenizers-cpp tokenizer stack. Those are an **internal build detail**, not the
`ort-server` release version — `ort-server` has its own semver (see
[VERSIONS.md](../VERSIONS.md)). This doc is the checklist for moving to a new
ORT release. It is written so a future automated (Claude Code) session can
execute it end-to-end.

## What the version knobs are

All live in [`CMakeLists.txt`](../CMakeLists.txt):

```cmake
set(ONNXRUNTIME_VERSION "1.27.0" ...)   # ORT prebuilt release (no leading v in the value)
set(_ort_sha256 "...")                  # per-platform SHA256 of the four ORT archives
# tokenizers-cpp / cpp-httplib / nlohmann-json are pinned by commit SHA
```

`ONNXRUNTIME_VERSION` templates the FetchContent download URL for the
per-platform prebuilt archive; each platform's `_ort_sha256` pins the exact
archive contents.

## Steps

1. **Pick the target ORT version** and confirm its four assets exist and still
   match the URL pattern in `CMakeLists.txt`
   (`onnxruntime-{win-x64,linux-x64,linux-aarch64,osx-arm64}-<ver>.{zip,tgz}`):
   ```
   gh api repos/microsoft/onnxruntime/releases/tags/v<X.Y.Z> \
     --jq '.tag_name, (.assets[].name | select(test("win-x64|linux-x64|linux-aarch64|osx-arm64")))'
   ```
   If Microsoft changed the naming, update the `_ort_pkg` logic.

2. **Bump `ONNXRUNTIME_VERSION` and recompute all four `_ort_sha256` pins**:
   download each archive and `sha256sum` it. A mismatched hash fails the
   configure step by design.

3. **Refresh the bundled ORT license texts** if upstream changed them — the
   release workflow stages `LICENSE` + `ThirdPartyNotices.txt` from the fetched
   package automatically, so normally nothing to do.

4. **Build all platforms** via the release workflow (dispatch, not a tag yet):
   ```
   gh workflow run release.yml -R lemonade-sdk/ort-server -f version=<next-ort-server-ver>
   gh run watch <run-id> -R lemonade-sdk/ort-server
   ```
   The workflow runs `test/smoke.py` (27+ contract checks) against the
   **extracted packaged archive** on every platform. Fix any compile/link
   breaks (ORT C++ API changes surface in `src/main.cpp`). Common gotchas are
   captured in git history (CMake-4/VS-2026, macOS `.dSYM` staging).

5. **Re-run the parity gate.** Run the catalog harness
   (`tools/classifier_catalog/` in this repo) against the new build to confirm
   the exported ONNX graphs still produce identical label scores. If an opset
   is too old for the new ORT, re-export that model.

6. **If — and only if — a target version is a hard blocker**, step down one ORT
   minor at a time until green, and record the blocker in the release notes.
   Prefer the latest ORT.

7. **Release.** Bump `ort-server`'s own version, update `VERSIONS.md`, tag it
   — the tag is the bare version, **NO leading `v`**: lemond's download URL is
   `releases/download/<pin>/ort-server-<pin>-<platform>...`, so the git tag
   must equal the version pin exactly:
   ```
   git tag <next-ort-server-ver> && git push origin <next-ort-server-ver>
   ```
   The tagged run rebuilds and attaches the artifacts to the GitHub Release.

8. **Point lemond at it.** In the lemonade repo, set
   `src/cpp/resources/backend_versions.json` -> `onnxruntime.cpu` to the new
   `ort-server` release version. lemond only ever references the `ort-server`
   release version, never the ORT version — this is the boundary that keeps
   ORT upgrades from leaking into lemond.

## Invariant

`ort-server` version ≠ ORT version. lemond tracks the `ort-server` release; the
bundled dependency versions are recorded in `VERSIONS.md`, and `GET /health`
reports the bundled ORT version (`{"status":"ok","onnxruntime":"<ver>"}`).
