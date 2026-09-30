# Zero-shot routing

`task: "zero-shot-classification"` serves a model that takes its **labels from
each request** instead of from a manifest. The caller sends a text and a list of
labels; the server returns a score for each.

```
POST /classify
{"text": "Write a Python function that merges two sorted lists.",
 "labels": ["code generation", "math reasoning", "creative writing"]}

{"labels": {"code generation": 0.71, "math reasoning": 0.19, "creative writing": 0.10}}
```

The reference model is <Model checkpoint. WIP>
which picks which LLM should answer a prompt — the label list is the set of
models you happen to have, so it cannot be baked in at export time.

## Why this is not just another classifier

A fixed-label classifier reads one sequence and emits one logit per label from
its head. This model has no per-label head. Instead the labels are written into
the text it reads:

```
Categories:
- code generation
- math reasoning
- creative writing

Text:
Write a Python function that merges two sorted lists.
```

and the model is told, separately, **which tokens are which** — via two pooling
grids supplied as ordinary model inputs:

| Input | Shape | Meaning |
|-------|-------|---------|
| `input_ids` | `[1, S]` | the glued block above |
| `attention_mask` | `[1, S]` | all ones |
| `text_pool` | `[1, 1, S]` | weights that average the tokens of the user's text |
| `category_pool` | `[1, R, S]` | row `r` averages the tokens of label `r` |

Each row is uniform over its member tokens (`1/k` for `k` members, `0`
elsewhere). The output is `[1, R]` — one logit per label — softmaxed to give the
scores. `R` is a genuine runtime dimension, which is the whole point: the export
must **not** freeze a label count into the graph.

## Building the grids

Membership is decided from the **character offsets** the tokenizer already
computes, matching the reference implementation:

* Record each label's character range while gluing the prompt. Label `r` starts
  2 characters after the preceding newline (skipping `"- "`) and runs for the
  length of the label.
* Tokenize the whole block once, with offsets.
* A token belongs to the **text** if its span ends past where the text starts.
* A token belongs to **label `r`** if its span overlaps label `r`'s range.
* A token with a **zero-width** span (`start == end`) belongs to neither. This
  is what excludes the inserted `<|startoftext|>` marker, which the tokenizer
  reports at `(0, 0)`.

## Model directory

```
router-dir/
  model.onnx             # 4 inputs -> logits [1, R]; R must be dynamic
  tokenizer.json         # with a truncation.max_length, unless the manifest sets one
  config.json            # model_type: , architectures:
  manifest.json          # OPTIONAL
```

`id2label` is **rejected** for this task (the labels come from the request, so a
baked-in list would be a false statement about the model), and
`score_normalization` must be `softmax` (one label set scored against one text).

Without a manifest the task is inferred from an `architectures` entry ending in
`ForSequenceRouting` or `ForZeroShotClassification`.

## Architecture allowlist

The zero-shot task has its **own** allowlist (`lfm2`), separate from the
encoder tasks'. The two conventions share nothing: the encoder path fabricates
an all-ones mask and all-zero `token_type_ids` and truncates by keeping the
trailing token; the router path has no segment ids, supplies explicit pooling
grids, and truncates with a plain cut. Both directions are refused — `lfm2`
under `text-classification` and `bert` under `zero-shot-classification` are
startup errors.

## Exporting

`torch.onnx.export` must keep all four of the model's inputs so that `R` stays a
runtime dimension:

```python
dynamic_axes={"input_ids": {0: "b", 1: "s"},
              "attention_mask": {0: "b", 1: "s"},
              "text_pool": {0: "b", 2: "s"},
              "category_pool": {0: "b", 1: "r", 2: "s"},
              "logits": {0: "b", 1: "r"}}
```

Exporting the convenience `route()` wrapper instead bakes the example's label
count into the graph; ort-server catches that at inference time (the output's
last dimension will not match the number of labels supplied) but the export is
useless either way.

## Tests

* `test/fixtures/make_tiny_router.py` builds `tiny-router`: a 17 KB fixture with
  a byte-level tokenizer and a graph whose logits are the dot product of the two
  pooled vectors, so a mis-built or zero-filled grid changes the answer. Its
  `golden.json` is produced by a transcription of `route()`, and `smoke.py`
  section J replays it.
* Sections J2/K/L/M/N cover the labels-vs-no-labels 400s both ways,
  manifest-less inference, both allowlist rejections, window overrun, and the
  manifest fields that contradict the task.
