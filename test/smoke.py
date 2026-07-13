"""Contract smoke tests for a built/packaged ort-server binary.

    python3 test/smoke.py <path-to-ort-server-binary>

Covers: softmax/sigmoid normalization, manifest-less config.json inference,
token-classification max/mean aggregation, truncation, top_k, request
validation (400s), startup rejection of bad manifests, and the output-dim
guard. Stdlib only, so it runs on any CI runner.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
PORT = 8231
BASE = f"http://127.0.0.1:{PORT}"
FAILURES = []


def request(payload, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}/classify", data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}
    except (urllib.error.URLError, OSError) as e:
        return 0, {"error": f"connection failed: {e}"}


class Server:
    def __init__(self, binary, model_dir):
        self.proc = subprocess.Popen(
            [binary, "--model-path", str(model_dir), "--port", str(PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def wait_ready(self, timeout=30):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(f"{BASE}/health", timeout=2):
                    return True
            except Exception:
                time.sleep(0.3)
        return False

    def stop(self):
        if self.proc.poll() is None:
            self.proc.kill()
        out = (
            self.proc.stdout.read().decode(errors="replace") if self.proc.stdout else ""
        )
        self.proc.wait()
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def variant(
    base_dir, tmp, name, manifest_edits=None, drop_manifest=False, config_edits=None
):
    d = Path(tmp) / name
    shutil.copytree(base_dir, d)
    if drop_manifest:
        (d / "manifest.json").unlink()
    if manifest_edits is not None:
        m = json.loads((d / "manifest.json").read_text())
        m.update(manifest_edits)
        (d / "manifest.json").write_text(json.dumps(m))
    if config_edits is not None:
        c = json.loads((d / "config.json").read_text())
        c.update(config_edits)
        (d / "config.json").write_text(json.dumps(c))
    return d


def scores_of(body):
    labels = body.get("labels", {})
    return labels if isinstance(labels, dict) else {}


GOLDEN_TOL = 1e-4


def check_golden(fixture_dir, label):
    """The oracle: ort-server must reproduce the HuggingFace reference scores.

    Structural checks pass even when tokenization is wrong (that is how the
    missing-special-tokens bug survived). These compare actual values against
    HF-tokenizer + ONNX Runtime output for the same inputs, including a
    truncated and a unicode case. Regenerate with fixtures/make_golden.py.
    """
    golden = json.loads((fixture_dir / "golden.json").read_text(encoding="utf-8"))
    worst = 0.0
    worst_case = ""
    for case in golden["cases"]:
        st, body = request({"text": case["text"]})
        if st != 200:
            check(f"{label}: golden request 200", False, f"status {st}: {body}")
            return
        got = scores_of(body)
        expected = case["labels"]
        if set(got) != set(expected):
            check(
                f"{label}: golden labels match",
                False,
                f"{sorted(got)} != {sorted(expected)}",
            )
            return
        for k, v in expected.items():
            delta = abs(got[k] - v)
            if delta > worst:
                worst, worst_case = delta, case["text"][:40]
    check(
        f"{label}: matches HuggingFace reference (max delta {worst:.2e})",
        worst <= GOLDEN_TOL,
        f"worst delta {worst:.3e} > {GOLDEN_TOL} on {worst_case!r}",
    )


def main():
    binary = str(Path(sys.argv[1]).resolve())
    clf = HERE / "fixtures" / "tiny-clf"
    tok = HERE / "fixtures" / "tiny-tok"
    tmp = tempfile.mkdtemp(prefix="ort-smoke-")

    # A: sequence classification, explicit manifest (softmax)
    with Server(binary, clf) as s:
        check("A: server ready (manifest)", s.wait_ready())
        st, body = request({"text": "hello world"})
        a_scores = scores_of(body)
        check("A: 200 with 2 labels", st == 200 and len(a_scores) == 2, str(body))
        check("A: scores in [0,1]", all(0.0 <= v <= 1.0 for v in a_scores.values()))
        check("A: softmax sums to 1", abs(sum(a_scores.values()) - 1.0) < 1e-3)
        st, body = request({"text": "hello world", "top_k": 1})
        check(
            "A: top_k=1 returns 1 label",
            st == 200 and len(scores_of(body)) == 1,
            str(body),
        )
        st, _ = request(None, raw=b"{not json")
        check("A: malformed JSON is 400", st == 400)
        st, _ = request({"text": 123})
        check("A: non-string text is 400", st == 400)
        st, _ = request({"top_k": 1})
        check("A: missing text/input is 400", st == 400)
        st, body = request({"input": "hello world"})
        check("A: 'input' alias works", st == 200 and len(scores_of(body)) == 2)
        check_golden(clf, "A: GOLDEN seq-cls")

    # A2: a tokenizer.json with PADDING enabled (as real HF repos ship) must
    # produce identical scores — the pad ids must never reach the model.
    pad = HERE / "fixtures" / "tiny-pad"
    with Server(binary, pad) as s:
        check("A2: padded-tokenizer server ready", s.wait_ready())
        st, body = request({"text": "hello world"})
        p_scores = scores_of(body)
        check(
            "A2: padded tokenizer gives identical scores",
            st == 200
            and all(abs(p_scores.get(k, -1) - v) < 1e-6 for k, v in a_scores.items()),
            f"{p_scores} != {a_scores}",
        )
        check_golden(pad, "A2: GOLDEN padded-tokenizer")

    # B: manifest-less — contract inferred from config.json
    with Server(binary, variant(clf, tmp, "noman", drop_manifest=True)) as s:
        check("B: server ready (config.json inference)", s.wait_ready())
        st, body = request({"text": "hello world"})
        b_scores = scores_of(body)
        check(
            "B: inferred contract matches manifest run",
            st == 200
            and all(abs(b_scores.get(k, -1) - v) < 1e-5 for k, v in a_scores.items()),
            str(body),
        )

    # C: sigmoid normalization
    with Server(
        binary, variant(clf, tmp, "sig", {"score_normalization": "sigmoid"})
    ) as s:
        check("C: server ready (sigmoid)", s.wait_ready())
        st, body = request({"text": "hello world"})
        c_scores = scores_of(body)
        check(
            "C: sigmoid scores in [0,1]",
            st == 200 and all(0.0 <= v <= 1.0 for v in c_scores.values()),
        )
        check(
            "C: sigmoid differs from softmax",
            any(abs(c_scores.get(k, 0) - v) > 1e-6 for k, v in a_scores.items()),
        )

    # D: truncation at manifest max_length
    with Server(binary, variant(clf, tmp, "trunc", {"max_length": 4})) as s:
        check("D: server ready (max_length=4)", s.wait_ready())
        st, body = request({"text": "word " * 5000})
        check(
            "D: over-length input classifies after truncation",
            st == 200 and len(scores_of(body)) == 2,
            str(body),
        )

    # E/F: token classification, max vs mean aggregation
    with Server(binary, tok) as s:
        check("E: token-cls server ready", s.wait_ready())
        st, body = request({"text": "hello world again"})
        e_scores = scores_of(body)
        check(
            "E: token-cls 3 labels in [0,1]",
            st == 200
            and len(e_scores) == 3
            and all(0.0 <= v <= 1.0 for v in e_scores.values()),
            str(body),
        )
        check_golden(tok, "E: GOLDEN token-cls")
    with Server(binary, variant(tok, tmp, "mean", {"token_aggregation": "mean"})) as s:
        check("F: token-cls mean ready", s.wait_ready())
        st, body = request({"text": "hello world again"})
        f_scores = scores_of(body)
        check(
            "F: mean differs from max",
            st == 200
            and any(abs(f_scores.get(k, 0) - v) > 1e-6 for k, v in e_scores.items()),
        )

    # G: token-cls manifest-less (task inferred from architectures)
    with Server(binary, variant(tok, tmp, "toknoman", drop_manifest=True)) as s:
        check("G: token-cls config.json inference ready", s.wait_ready())
        st, body = request({"text": "hello world again"})
        check(
            "G: inferred token-cls matches manifest run",
            st == 200
            and all(
                abs(scores_of(body).get(k, -1) - v) < 1e-5 for k, v in e_scores.items()
            ),
        )

    # H: invalid manifests are startup errors
    for i, (edits, name) in enumerate(
        [
            ({"score_normalization": "none"}, "H: score_normalization 'none' rejected"),
            ({"task": "image-classification"}, "H: unknown task rejected"),
            (
                {"token_aggregation": "first-subword", "task": "token-classification"},
                "H: unknown token_aggregation rejected",
            ),
        ]
    ):
        with Server(binary, variant(clf, tmp, f"bad{i}", edits)) as s:
            ready = s.wait_ready(timeout=10)
            out = s.stop()
            check(name, not ready, out[-200:])

    # H2: an architecture whose mask/segment conventions we don't implement
    # (XLNet et al.) must be refused, not silently mis-served.
    with Server(
        binary, variant(clf, tmp, "xlnet", config_edits={"model_type": "xlnet"})
    ) as s:
        ready = s.wait_ready(timeout=10)
        out = s.stop()
        check("H: unsupported model_type rejected", not ready, out[-200:])

    # H3: a truncated tokenizer.json must be a clean startup error, not a Rust
    # panic across the FFI (which aborts with no usable message).
    corrupt = variant(clf, tmp, "corrupttok")
    (corrupt / "tokenizer.json").write_text("")
    with Server(binary, corrupt) as s:
        ready = s.wait_ready(timeout=10)
        out = s.stop()
        check(
            "H: corrupt tokenizer.json reports a clean error",
            not ready and "not valid JSON" in out,
            out[-200:],
        )

    # I: model output dim vs id2label mismatch is a clean 500
    with Server(
        binary, variant(clf, tmp, "dim", {"id2label": {"0": "A", "1": "B", "2": "C"}})
    ) as s:
        check("I: dim-mismatch server ready", s.wait_ready())
        st, body = request({"text": "hi"})
        check("I: mismatch is 500 with error", st == 500 and "error" in body, str(body))

    shutil.rmtree(tmp, ignore_errors=True)
    if FAILURES:
        print(f"\n{len(FAILURES)} smoke check(s) FAILED: {FAILURES}")
        sys.exit(1)
    print("\nall smoke checks passed")


if __name__ == "__main__":
    main()
