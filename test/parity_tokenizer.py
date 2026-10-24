"""Compare `third_party/tok_ffi` with the HuggingFace reference, token for token.

This is the harness behind the 20,001-case run in PR #4. Re-run it whenever the
tokenizer moves: a crate bump, a change to `tok_encode`, or a new tokenizer
family in the catalog.

    python test/parity_tokenizer.py test/fixtures/tiny-router
    python test/parity_tokenizer.py <model-dir> -n 20001
    python test/parity_tokenizer.py <model-dir> --cases corpus.txt

What it compares, per input: the token ids, the per-token offset STARTS and the
per-token offset ENDS. All three must match exactly. Scores are not involved --
this isolates the tokenizer, so a failure here has exactly one cause.

Why offsets and not just ids: ids alone pass even when the offsets are reported
in bytes instead of characters, and the zero-shot router decides which tokens
belong to which label from those offsets alone. A byte-offset build serves
confident, wrong routes. `offsets_are_character_indexed` in src/lib.rs pins one
case of this; this harness pins it across a corpus.

Needs `transformers` (preferred) or `tokenizers`, and cargo. None of these ship
with the server, so this is a manual gate, not a release-job step.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "third_party" / "tok_ffi" / "Cargo.toml"

# Fragments chosen for the ways a tokenizer offset can go wrong. Each line is a
# property, not just a string.
FRAGMENTS = [
    "hello world",                       # plain ASCII
    "The quick brown fox jumps over.",   # ASCII with punctuation
    "café naïve façade",                 # Latin-1 accents: 2 bytes, 1 character
    "déjà vu",                           # combining-capable accents
    "🚀 launch 🛰 orbit",                 # astral plane: 4 bytes, 1 character
    "égalité",               # combining acute: 2 code points, 1 glyph
    "日本語のテキスト",                      # CJK: 3 bytes per character
    "Здравствуй, мир",                   # Cyrillic
    "مرحبا بالعالم",                      # RTL
    "ﬁ ligature and ​ zero width",  # ligature + zero-width space
    "   leading and trailing   ",        # whitespace is part of a span
    "tabs\tand\nnewlines",               # control characters in the TEXT (legal)
    "C++ / rust-lang @ 3.14 #hash",      # punctuation runs
    "1234567890" * 8,                    # long digit run
    "",                                  # empty
    " ",                                 # whitespace only
]

# The router feeds this exact shape, and its label ranges are character ranges
# into it. A corpus of bare sentences would never exercise it.
LABEL_POOL = [
    "code generation", "math reasoning", "creative writing", "café",
    "🚀 launch", "C++", "small talk", "summarisation", "日本語", "plain",
]


def build_corpus(n: int, seed: int) -> list[str]:
    """A deterministic corpus: same seed, same cases, so a failure is repeatable."""
    rng = random.Random(seed)
    cases: list[str] = list(FRAGMENTS)
    while len(cases) < n:
        kind = rng.random()
        if kind < 0.45:
            body = " ".join(rng.choice(FRAGMENTS) for _ in range(rng.randint(1, 6)))
            cases.append(body)
        elif kind < 0.9:
            labels = rng.sample(LABEL_POOL, rng.randint(1, 5))
            text = " ".join(rng.choice(FRAGMENTS) for _ in range(rng.randint(1, 4)))
            block = "\n".join(f"- {l}" for l in labels)
            cases.append(f"Categories:\n{block}\n\nText:\n{text}")
        else:
            # Longer than any window, so the tail of a long input is covered.
            cases.append(" ".join(rng.choice(FRAGMENTS) for _ in range(200)))
    return cases[:n]


def load_cases(path: Path) -> list[str]:
    """One case per line. A .jsonl line is a JSON string; a .txt line is raw."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in lines if l.strip()]
    return lines


def fixed_padding(tokenizer_json: Path) -> int | None:
    """The fixed width tokenizer.json pads to, or None.

    This matters because tok_new treats the two settings in the file
    DIFFERENTLY, on purpose:

      * truncation is CLEARED, so that the token budget has exactly one owner,
        ort-server's max_length;
      * padding is KEPT, because a padded tokenizer is a real family and
        ort-server filters the pad tokens downstream (fixture tiny-pad; the
        0.3.1 regression that served wrong scores).

    A reference that drops both would report every padded tokenizer as a
    mismatch, which says nothing about the FFI layer.
    """
    try:
        pad = json.loads(tokenizer_json.read_text(encoding="utf-8")).get("padding")
    except (OSError, ValueError):
        return None
    strategy = (pad or {}).get("strategy")
    if isinstance(strategy, dict) and isinstance(strategy.get("Fixed"), int):
        return strategy["Fixed"]
    # "BatchLongest" pads a batch to its longest member, which is a no-op for
    # the one-sequence-at-a-time encoding both sides do here.
    return None


def reference_encoder(model_dir: Path, add_special: bool):
    """Return (name, fn) where fn(text) -> (ids, starts, ends).

    `transformers` first: that is the pipeline this server claims parity with.
    """
    pad_to = fixed_padding(model_dir / "tokenizer.json")
    try:
        from transformers import AutoTokenizer
        from transformers.utils import logging as hf_logging

        # A long case is expected here and is not an error: this harness never
        # runs a model, so "longer than the maximum sequence length" is noise.
        hf_logging.set_verbosity_error()

        tok = AutoTokenizer.from_pretrained(str(model_dir))
        if not tok.is_fast:
            raise RuntimeError("not a fast tokenizer")

        pad_kwargs = (
            {"padding": "max_length", "max_length": pad_to}
            if pad_to
            else {"padding": False}
        )

        def encode(text: str):
            enc = tok(
                text,
                add_special_tokens=add_special,
                truncation=False,
                return_offsets_mapping=True,
                **pad_kwargs,
            )
            offs = enc["offset_mapping"]
            return (list(enc["input_ids"]),
                    [o[0] for o in offs],
                    [o[1] for o in offs])

        return f"transformers (pad_to={pad_to})", encode
    except Exception as e:  # noqa: BLE001 - any failure falls through
        first = e

    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tok.no_truncation()  # as tok_new does; padding is left as the file set it

        def encode(text: str):
            enc = tok.encode(text, add_special_tokens=add_special)
            return (list(enc.ids),
                    [o[0] for o in enc.offsets],
                    [o[1] for o in enc.offsets])

        return f"tokenizers (pad_to={pad_to})", encode
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"no reference tokenizer available.\n"
            f"  transformers: {first}\n  tokenizers:   {e}\n"
            f"Install one: pip install transformers"
        )


def build_dump(target_dir: Path) -> Path:
    print("building the dump example (cargo)...", file=sys.stderr)
    subprocess.run(
        ["cargo", "build", "--release", "--locked",
         "--manifest-path", str(MANIFEST),
         "--target-dir", str(target_dir), "--example", "dump"],
        check=True,
    )
    for name in ("dump.exe", "dump"):
        p = target_dir / "release" / "examples" / name
        if p.exists():
            return p
    raise SystemExit(f"cargo succeeded but no dump binary under {target_dir}")


def run_ffi(dump: Path, tokenizer_json: Path, cases: list[str], add_special: bool):
    """Feed every case to the dump binary and yield its answers in order."""
    argv = [str(dump), str(tokenizer_json)]
    if not add_special:
        argv.append("--no-special")
    proc = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0
    )

    def feed():
        try:
            for c in cases:
                b = c.encode("utf-8")
                proc.stdin.write(f"{len(b)}\n".encode())
                proc.stdin.write(b)
                proc.stdin.write(b"\n")
            proc.stdin.close()
        except BrokenPipeError:
            pass  # the child died; the read side reports the real error

    # A writer thread, because 20k records do not fit in a pipe buffer: writing
    # them all before reading any would deadlock.
    t = threading.Thread(target=feed, daemon=True)
    t.start()
    for _ in cases:
        line = proc.stdout.readline()
        if not line:
            proc.wait()
            raise SystemExit(
                f"dump exited early (code {proc.returncode}) with cases left"
            )
        yield json.loads(line)
    proc.stdout.close()
    proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="directory holding tokenizer.json")
    ap.add_argument("-n", type=int, default=20001, help="generated case count")
    ap.add_argument("--seed", type=int, default=20001)
    ap.add_argument("--cases", type=Path, help="corpus file, one case per line")
    ap.add_argument("--dump", type=Path, help="prebuilt dump binary")
    ap.add_argument("--target-dir", type=Path, default=ROOT / "build" / "tok_ffi_parity")
    ap.add_argument("--no-special", action="store_true",
                    help="encode without the post-processor's inserted markers")
    ap.add_argument("--max-report", type=int, default=5)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer_json = model_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        raise SystemExit(f"no tokenizer.json in {model_dir}")

    add_special = not args.no_special
    cases = load_cases(args.cases) if args.cases else build_corpus(args.n, args.seed)
    ref_name, encode_ref = reference_encoder(model_dir, add_special)
    dump = args.dump or build_dump(args.target_dir)

    print(
        f"{len(cases)} cases | reference: {ref_name} | "
        f"add_special_tokens={add_special} | {tokenizer_json}",
        file=sys.stderr,
    )

    failures = 0
    reported = 0
    tokens = 0
    for i, (case, got) in enumerate(
        zip(cases, run_ffi(dump, tokenizer_json, cases, add_special))
    ):
        if "error" in got:
            failures += 1
            if reported < args.max_report:
                reported += 1
                print(f"FAIL case {i}: tok_encode failed on {case[:60]!r}")
            continue
        want_ids, want_starts, want_ends = encode_ref(case)
        tokens += len(want_ids)
        diffs = [
            name
            for name, w, g in (
                ("ids", want_ids, got["ids"]),
                ("starts", want_starts, got["starts"]),
                ("ends", want_ends, got["ends"]),
            )
            if list(w) != list(g)
        ]
        if not diffs:
            continue
        failures += 1
        if reported < args.max_report:
            reported += 1
            print(f"FAIL case {i} ({', '.join(diffs)}): {case[:80]!r}")
            for name, w, g in (
                ("ids", want_ids, got["ids"]),
                ("starts", want_starts, got["starts"]),
                ("ends", want_ends, got["ends"]),
            ):
                if list(w) == list(g):
                    continue
                at = next(
                    (k for k in range(min(len(w), len(g))) if w[k] != g[k]),
                    min(len(w), len(g)),
                )
                print(f"    {name}: first difference at token {at}")
                print(f"      reference: {list(w)[max(0, at - 2):at + 3]}")
                print(f"      tok_ffi:   {list(g)[max(0, at - 2):at + 3]}")
                if name in ("starts", "ends") and len(w) == len(g):
                    # The signature of a byte-offset build: the FFI side runs
                    # ahead of the reference on every multi-byte character.
                    print("      (offsets differ: check encode_char_offsets)")

    print(
        f"\n{len(cases) - failures}/{len(cases)} cases match "
        f"({tokens} tokens compared)"
    )
    if failures:
        print(f"{failures} FAILED")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
