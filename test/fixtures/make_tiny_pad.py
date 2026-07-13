"""Generate tiny-pad: tiny-clf whose tokenizer.json enables PADDING.

Real HuggingFace repos ship tokenizer.json files with a `padding` section (the
published phishing classifier does). The Rust tokenizer then pads every encoding
out to a fixed width, while `transformers` does not pad by default — so a server
that forwards the padded ids under an all-ones attention mask makes the model
attend to [PAD] as if it were text, and the scores silently diverge.

Scores here must be IDENTICAL to tiny-clf's: same model, same text, padding is
supposed to be invisible.

    conda run -n lmxclf python test/fixtures/make_tiny_pad.py
"""

import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
src = HERE / "tiny-clf"
dst = HERE / "tiny-pad"

if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
(dst / "golden.json").unlink(missing_ok=True)

tj = json.loads((dst / "tokenizer.json").read_text(encoding="utf-8"))
tj["padding"] = {
    "strategy": {"Fixed": 64},
    "direction": "Right",
    "pad_to_multiple_of": None,
    "pad_id": 0,
    "pad_type_id": 0,
    "pad_token": "[PAD]",
}
(dst / "tokenizer.json").write_text(json.dumps(tj, indent=2), encoding="utf-8")
print("tiny-pad written (padding: Fixed 64, pad_id 0)")
