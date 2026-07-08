# Upgrading the ONNX Runtime version

`ort-server` bundles a specific ONNX Runtime (ORT) + onnxruntime-extensions.
Those are an **internal build detail**, not the `ort-server` release version —
`ort-server` has its own semver (see [VERSIONS.md](../VERSIONS.md)). This doc is
the checklist for moving to a new ORT release. It is written so a future
automated (Claude Code) session can execute it end-to-end.

## What the version knobs are

Both live at the top of [`CMakeLists.txt`](../CMakeLists.txt):

```cmake
set(ONNXRUNTIME_VERSION "1.27.0" ...)   # ORT prebuilt release (no leading v in the value)
set(ORT_EXTENSIONS_TAG  "v0.14.0" ...)  # onnxruntime-extensions git tag (leading v)
```

`ONNXRUNTIME_VERSION` templates the FetchContent download URL for the per-platform
prebuilt archive. `ORT_EXTENSIONS_TAG` is the git tag built from source for the
in-graph tokenizers. **These two must be compatible** — a new ORT often needs a
newer extensions tag.

## Steps

1. **Pick the target versions.**
   - Latest ORT release + its exact asset naming:
     ```
     gh api repos/microsoft/onnxruntime/releases/tags/v<X.Y.Z> \
       --jq '.tag_name, (.assets[].name | select(test("win-x64|linux-x64|linux-aarch64|osx-arm64")))'
     ```
     Confirm the four assets exist and still match the URL pattern in `CMakeLists.txt`
     (`onnxruntime-{win-x64,linux-x64,linux-aarch64,osx-arm64}-<ver>.{zip,tgz}`). If
     Microsoft changed the naming, update the `_ort_pkg` logic in `CMakeLists.txt`.
   - Latest compatible extensions tag:
     ```
     gh api repos/microsoft/onnxruntime-extensions/releases --jq '.[].tag_name'
     ```
     Use the newest tag released on/after the target ORT. If unsure, pick the newest.

2. **Bump both pins** in `CMakeLists.txt` (`ONNXRUNTIME_VERSION`, `ORT_EXTENSIONS_TAG`).

3. **Build all platforms** via the release workflow (dispatch, not a tag yet):
   ```
   gh workflow run release.yml -R lemonade-sdk/ort-server -f version=<next-ort-server-ver>
   gh run watch <run-id> -R lemonade-sdk/ort-server
   ```
   Fix any compile/link breaks (ORT C++ API changes surface in `src/main.cpp`;
   extensions registration/ABI changes surface at link). Common gotchas are
   captured in git history (CMake-4/VS-2026, opencv trim, string-tensor API).

4. **Re-run the parity gate.** In the lemonade repo, run the catalog harness
   (`tools/classifier_catalog/`) against the new build to confirm the exported
   ONNX graphs (opset) + tokenizer ops still produce identical label scores.
   ORT bumps go through the SAME validation as everything else. If an opset is
   too old for the new ORT, re-export that model.

5. **If — and only if — a target version is a hard blocker**, step down one ORT
   minor at a time until green, and record the blocker in the release notes.
   Prefer the latest ORT.

6. **Release.** Bump `ort-server`'s own version, update `VERSIONS.md`, tag it:
   ```
   git tag v<next-ort-server-ver> && git push origin v<next-ort-server-ver>
   ```
   The tagged run rebuilds and attaches the artifacts to the GitHub Release.

7. **Point lemond at it.** In the lemonade repo, set
   `src/cpp/resources/backend_versions.json` -> `onnxruntime.cpu` to the new
   `ort-server` release version (the value WITHOUT a leading `v`, matching the
   artifact names). lemond only ever references the `ort-server` release version,
   never the ORT version — this is the boundary that keeps ORT upgrades from
   leaking into lemond.

## Invariant

`ort-server` version ≠ ORT version. lemond tracks the `ort-server` release; the
bundled ORT/extensions versions are recorded in `VERSIONS.md` and reported by
`GET /health`.
