"""Generate golden.json: the HuggingFace reference scores each fixture must reproduce.

This is the oracle for ort-server's C++ path. It tokenizes with the *real* HF
tokenizer (special tokens on, truncation at the manifest's budget), runs the
same model.onnx through ONNX Runtime, and applies the manifest's normalization
and token aggregation — i.e. exactly what ort-server claims to do. smoke.py
then asserts ort-server's HTTP responses match these scores.

Structural checks (labels present, scores in [0,1]) cannot catch a tokenization
bug; this can. Re-run after any tokenization or contract change:

    conda run -n lmxclf python test/fixtures/make_golden.py
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

HERE = Path(__file__).parent
FIXTURES = ["tiny-clf", "tiny-tok"]

# Short, long (forces truncation), and unicode/punctuation-heavy inputs.
TEXTS = [
    "hello world",
    "Please verify your account at http://secure-login.example now.",
    "word " * 3000,
    "Ünïcödé — em-dash, quotes “x”, emoji 🎯, and\ttabs.",
]


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    for name in FIXTURES:
        d = HERE / name
        manifest = json.loads((d / "manifest.json").read_text())
        id2label = {int(k): v for k, v in manifest["id2label"].items()}
        normalize = sigmoid if manifest["score_normalization"] == "sigmoid" else softmax
        max_length = manifest.get("max_length", 512)
        token_agg = manifest.get("token_aggregation") or "max"

        tok = AutoTokenizer.from_pretrained(d)
        sess = ort.InferenceSession(
            str(d / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        in_names = {i.name for i in sess.get_inputs()}

        cases = []
        for text in TEXTS:
            enc = tok(
                text,
                return_tensors="np",
                truncation=True,
                max_length=max_length,
                return_special_tokens_mask=True,
            )
            special_mask = enc.pop("special_tokens_mask")[0].astype(bool)
            feeds = {k: v for k, v in enc.items() if k in in_names}
            logits = sess.run(None, feeds)[0][0]
            if manifest["task"] == "token-classification":
                probs = normalize(logits)  # [tokens, labels]
                # The HuggingFace token-classification pipeline drops special
                # tokens ([CLS]/[SEP]) from its output, so they must not be
                # aggregated here either.
                content = probs[~special_mask]
                agg = (
                    content.max(axis=0) if token_agg == "max" else content.mean(axis=0)
                )
            else:
                agg = normalize(logits)
            cases.append(
                {
                    "text": text,
                    "labels": {id2label[i]: float(agg[i]) for i in range(len(agg))},
                }
            )

        golden = {
            "generated_from": "HuggingFace tokenizer + ONNX Runtime (python)",
            "task": manifest["task"],
            "score_normalization": manifest["score_normalization"],
            "max_length": max_length,
            "cases": cases,
        }
        (d / "golden.json").write_text(json.dumps(golden, indent=2))
        print(f"{name}: wrote golden.json ({len(cases)} cases)")


if __name__ == "__main__":
    main()
