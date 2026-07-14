"""Fail if the packaged binary depends on anything we do not ship.

ort-server is distributed as a self-contained archive (binary + the ONNX Runtime
library). If a build machine happens to have OpenSSL/zlib/etc. installed, CMake
dependencies can silently link against them; the binary then works on the build
machine and dies with STATUS_DLL_NOT_FOUND (or a missing .so) on hosts without
them. That is exactly how a broken archive shipped once, so the allowlist below
is enforced on every release.

    python3 test/check_deps.py <path-to-ort-server-binary>
"""

import subprocess
import sys
from pathlib import Path

# Bundled with the binary, or guaranteed present on a stock OS install.
WINDOWS_ALLOWED = {
    "onnxruntime.dll",  # shipped in the archive
    "kernel32.dll",
    "ntdll.dll",
    "ws2_32.dll",
    "wsock32.dll",
    "advapi32.dll",
    "bcryptprimitives.dll",
    "bcrypt.dll",
    "user32.dll",
    "ole32.dll",
    "oleaut32.dll",
    "shell32.dll",
    "crypt32.dll",
    "dbghelp.dll",
    "userenv.dll",
    "secur32.dll",
    "msvcp140.dll",  # MSVC redistributable
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcrt.dll",
    "psapi.dll",
    "powrprof.dll",
    "pdh.dll",
}

UNIX_ALLOWED_PREFIXES = (
    "libonnxruntime",  # shipped in the archive
    "libc.",
    "libm.",
    "libdl.",
    "librt.",
    "libpthread.",
    "libgcc_s.",
    "libstdc++.",
    "ld-linux",
    "linux-vdso",
    "libSystem.",  # macOS
    "libc++.",
)

# Frameworks that ship with macOS itself.
MACOS_ALLOWED_PATH_PREFIXES = ("/System/Library/Frameworks/", "/usr/lib/")


def windows_deps(binary):
    import pefile  # only needed on Windows runners

    pe = pefile.PE(binary, fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
    )
    return sorted(e.dll.decode().lower() for e in pe.DIRECTORY_ENTRY_IMPORT)


def unix_deps(binary):
    tool = ["otool", "-L"] if sys.platform == "darwin" else ["ldd"]
    out = subprocess.run(tool + [binary], capture_output=True, text=True).stdout
    deps = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        name = line.split()[0]
        deps.append(name if sys.platform == "darwin" else Path(name).name)
    return sorted(deps)


def main():
    binary = sys.argv[1]
    if sys.platform == "win32":
        deps = windows_deps(binary)
        bad = [
            d
            for d in deps
            if d not in WINDOWS_ALLOWED and not d.startswith("api-ms-win-")
        ]
    else:
        deps = unix_deps(binary)

        def allowed(d):
            if any(Path(d).name.startswith(p) for p in UNIX_ALLOWED_PREFIXES):
                return True
            if sys.platform == "darwin":
                return d.startswith(MACOS_ALLOWED_PATH_PREFIXES)
            return False

        bad = [d for d in deps if not allowed(d)]

    print("dependencies:")
    for d in deps:
        print(f"  {d}")
    if bad:
        print(
            "\nFAIL: the binary depends on libraries that are neither bundled nor "
            "guaranteed on a stock OS:"
        )
        for d in bad:
            print(f"  {d}")
        print("\nThese resolve on the build machine and fail everywhere else.")
        sys.exit(1)
    print("\nOK: binary is self-contained")


if __name__ == "__main__":
    main()
