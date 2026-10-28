# Zero-shot classification

`task: "zero-shot-classification"` serves a model that takes its **labels from
each request** instead of from a manifest.

```
POST /classify
{"text": "Write a Python function that merges two sorted lists.",
 "labels": ["code generation", "math reasoning", "creative writing"]}

{"labels": {"code generation": 0.71, "math reasoning": 0.19, "creative writing": 0.10}}
```

Scores are softmaxed over the supplied list, so they sum to 1 before `top_k`.
A typical use is picking which model should answer a prompt: the candidate set
differs per deployment, so it cannot be baked in at export time.

## Model

[`lemonade-sdk/LFM2.5-Encoder-350M-Prompt-Router-ONNX`](https://huggingface.co/lemonade-sdk/LFM2.5-Encoder-350M-Prompt-Router-ONNX)
is a ready-to-serve model directory. It is an ONNX export of
[`LiquidAI/LFM2.5-Encoder-350M-Prompt-Router`](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Prompt-Router),
made as in [Exporting a model](#exporting-a-model). Export it yourself only to
pin another revision of the original.

## How it works

A fixed-label classifier emits one logit per label from its head. A zero-shot
model has no per-label head. The labels are written into the text it reads:

```
Categories:
- code generation
- math reasoning
- creative writing

Text:
Write a Python function that merges two sorted lists.
```

Two pooling grids, supplied as ordinary model inputs, say which tokens are
which:

| Input | Shape | Meaning |
|-------|-------|---------|
| `input_ids` | `[1, S]` | the block above |
| `attention_mask` | `[1, S]` | all ones |
| `text_pool` | `[1, 1, S]` | averages the tokens of the text |
| `category_pool` | `[1, R, S]` | row `r` averages the tokens of label `r` |

Each row is uniform over its members: `1/k` for `k` members, `0` elsewhere. The
output is `[1, R]`. `R` is a runtime dimension, not a fixed label count.

Membership comes from the **character offsets** the tokenizer reports:

* Record each label's character range while building the block. Label `r` starts
  2 characters past the preceding newline, skipping `"- "`.
* Tokenize the block once, with offsets.
* A token belongs to the **text** if its span ends past where the text starts.
* A token belongs to **label `r`** if its span overlaps label `r`'s range.
* A **zero-width** span (`start == end`) belongs to neither. This excludes a
  start-of-sequence marker the tokenizer inserts, which reports `(0, 0)`.

The two tests are independent, so a token straddling the `"\n\nText:\n"` header
counts for both.

## Model directory

```
router-dir/
  model.onnx             # 4 inputs -> logits [1, R]; R must be dynamic
  tokenizer.json         # with a truncation.max_length, unless the manifest sets one
  config.json            # model_type and architectures
  manifest.json          # OPTIONAL
```

```json
{
  "task": "zero-shot-classification",
  "max_length": 8192,
  "max_labels": 2048,
  "max_label_chars": 8192,
  "max_grid_cells": 33554432
}
```

Omit any of the last three to take the derived default. `id2label` is rejected,
because the labels come from the request. `score_normalization` must be
`softmax`. Both are startup errors.

Without a manifest the task is inferred from an `architectures` entry ending in
`ForSequenceRouting` or `ForZeroShotClassification`.

## Architecture allowlist

This task has its **own** allowlist, separate from the encoder tasks'. It
currently holds `lfm2`. The two conventions share nothing: the encoder path
fabricates an all-ones mask and all-zero `token_type_ids` and truncates by
keeping the trailing token, while this path has no segment ids and supplies
explicit pooling grids. Both directions are refused, so `lfm2` under
`text-classification` and `bert` under `zero-shot-classification` are startup
errors.

## Exporting a model

```
pip install torch transformers onnx onnxruntime huggingface_hub
python tools/zero_shot_router/export.py <out_dir> --revision <commit>
python tools/zero_shot_router/parity.py <out_dir> --revision <commit> --binary <ort-server>
```

The script:

1. Loads the checkpoint with `trust_remote_code=True`. This is **required**:
   `Lfm2BidirForSequenceRouting` and `route()` are in the checkpoint's
   `modeling_lfm2_bidirectional.py`, not in `transformers`. Read that file at the
   commit you pin with `--revision`.
2. Replaces the short-conv forward. The checkpoint's version reads its padding
   and groups from tensor shapes. Tracing turns those into tensors, and
   `torch.onnx.export` fails with `expected padding to be a single integer
   value ... got padding=[]`.
3. Exports `forward()`, not `route()`. `route()` builds the pooling grids from
   its own arguments, so exporting it freezes the label count of the example.
4. Compares the graph with `route()` on 16 cases with 2 to 8 labels. It stops if
   a score differs by more than `1e-4` or a top label changes.
5. Writes the model directory, a `manifest.json` with `max_length` (default
   `max_position_embeddings`, or `--max-length`), and `golden.json`.

**The tokenizer must be a fast one.** The pooling grids are built from character
offsets, which a slow tokenizer does not report.

**Write `max_length` into `manifest.json`.** It decides whether a label set
fits, and a text cut to a narrow window scores differently from the same text
served whole. Without a manifest the server falls back to the `truncation` block
in `tokenizer.json`, which may carry a number chosen for training rather than
for serving.

## The token budget

A label set that does not fit is refused, never cut, because a label pooled from
half its tokens would score something the model only partly read. The text is
elastic, so it is cut and the cut is reported.

Three checks run, cheapest first:

| # | Check | Cost | Refuses |
|---|-------|------|---------|
| 1 | `max_labels`, `max_label_chars` | one comparison | counts no window could hold |
| 2 | label block, by characters | one division | a block no window could hold |
| 3 | label block, by tokens | one tokenization of the block | the real overrun |

Check 2 uses the longest token in the vocabulary: `C` characters need at least
`C / longest` tokens, and `budget × longest` characters is all a budget of
`budget` tokens can cover. Both are worst cases, so neither refuses or cuts
anything that would have fitted. A 5 MB text costs one pass over its characters,
not a 5 MB tokenization.

Check 3 is the window check. Its `400` gives both numbers:

```
the label list needs 725 tokens, and this model's window holds 512
with 10 reserved for markers and text. Send fewer or shorter labels.
```

## Request limits

**Command line beats manifest beats derived default.**

| Manifest field | CLI flag | Default | Meaning |
|----------------|----------|---------|---------|
| `max_labels` | `--max-labels` | `max_length / 4` | Labels per request |
| `max_label_chars` | `--max-label-chars` | `max_length` | Characters in one label |
| `max_grid_cells` | `--max-grid-cells` | `33554432` (32M, = 128 MB) | Ceiling on `labels × tokens` |

The first two defaults are derived, not chosen. A label costs at least 4 tokens
(`"- "`, one character, `"\n"`), so `max_length / 4` is the most any window could
hold, and one label cannot exceed the whole window. Nothing guards their
product, so passing both proves nothing; the token budget above is what decides.
Raising `max_labels` cannot make a request fit. Lowering it, to bound latency,
is the useful direction.

`max_grid_cells` is the only limit that bounds **memory**. `category_pool` holds
`labels × tokens` floats and both factors grow with `max_length`, so the product
grows as `max_length²`. A wide-context model reaches tens of GB with both
per-axis limits satisfied.

All three are startup errors on a non-router model, in either form, as is a
malformed flag value.

Three rules are fixed, and each answers `400`: control characters in a label,
duplicate labels, and an empty or whitespace-only label.

### Setting limits from Lemonade

`manifest.json` ships inside the model directory, and a re-download overwrites
it, so a Lemonade user cannot edit it durably. The CLI flags exist for that
case: Lemonade forwards any flag that is not `--model-path` or `--port` into
ort-server's argv.

```
lemonade load my-router --onnxruntime-args "--max-labels 4096"
```

To make it stick, save it in `recipe_options.json` in the Lemonade config
directory, keyed by model name with its source prefix:

```json
{"user.my-router": {"onnxruntime_args": "--max-labels 4096"}}
```

## Truncation

Only the text is ever cut. The response then carries one extra field:

```
{"labels": {...}, "truncated": true}
```

The field is absent when the whole input fitted. It is a boolean, never `0`/`1`,
so a score parser cannot read a stray numeric member as a label score.

**The cut keeps both ends.** A long message often carries its question at the
end, such as code pasted above *"why does this crash?"*. The text gets one third
of its budget from the head and two thirds from the tail, with `\n...\n` between
them so the model can see that something is missing.

The model reads the marker, but `text_pool` leaves it out: the marker is the
server's text, not the user's. The rule goes by position, so the same five
characters typed by the user are pooled as usual.

**The cut lands between characters.** One character can be several tokens, as
an emoji often is. A character the budget cannot afford whole is dropped whole.
When not one character fits on either side of the marker, the marker is dropped
and the tail takes the whole budget.

With a cut, the prompt no longer matches the reference implementation, which
has no budget and never cuts. Parity holds for input that fits.

Run with `--verbose` to see each cut on stderr:

```
[ort-server] text truncated: 3 labels cost 73 tokens of a 192-token window,
             leaving 118 for a text that needed 3776
```

## Tests

`test/fixtures/make_tiny_router.py` builds `tiny-router`: a 17 KB fixture whose
logits are the dot product of the two pooled vectors, so a mis-built or
zero-filled grid changes the answer.

`test/smoke.py` sections J through N cover this task: the golden, the
labels-vs-no-labels `400`s both ways, manifest-less inference, both allowlist
rejections, the window check, the three limits in manifest and CLI form, and
head+tail truncation. Each section's comment says what it pins.

The real model is not in CI. To check it, download it, or export it, then run
section R:

```
huggingface-cli download lemonade-sdk/LFM2.5-Encoder-350M-Prompt-Router-ONNX --local-dir <dir>
python test/smoke.py <binary> --router-dir <dir>
```

Section R compares the server with
`test/golden/LFM2.5-Encoder-350M-Prompt-Router.json`: `route()` scores from
the checkpoint at the revision the file names.
