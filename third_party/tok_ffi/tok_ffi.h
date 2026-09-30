// C ABI over the HuggingFace `tokenizers` crate. Implemented in src/lib.rs;
// these declarations must stay field-for-field in step with the #[repr(C)]
// struct there.
//
// Contract: every successful tok_encode must be matched by exactly one
// tok_encoding_free, and every non-null tok_new by one tok_free.

#ifndef ORT_SERVER_TOK_FFI_H
#define ORT_SERVER_TOK_FFI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TokHandle TokHandle;  // opaque

// Offsets are CHARACTER indices into the input, never byte indices. That is
// the unit the Python reference works in, and it is the unit that cannot slice
// a multi-byte character in half: a byte-level BPE splits one emoji across
// three tokens, and all three report the same character span, so a span
// touching any of them covers the whole character.
//
// The flip side is that character offsets are lossy for such a split: those
// three tokens are indistinguishable by span alone. Harmless for scoring and
// for redaction (both want the whole character), but per-token byte
// attribution cannot be recovered from this struct -- re-encode if you need it.
typedef struct {
    uint32_t* ids;     // token ids
    uint32_t* starts;  // per-token offset start, in CHARACTERS of the input
    uint32_t* ends;    // per-token offset end, in CHARACTERS of the input
    size_t len;        // element count of all three arrays
} TokEncoding;

// Returns NULL if the bytes are not a loadable tokenizer.json.
TokHandle* tok_new(const uint8_t* json, size_t json_len);

// Returns false on failure; `out` is only written on success.
bool tok_encode(const TokHandle* handle, const uint8_t* text, size_t text_len,
                bool add_special_tokens, TokEncoding* out);

void tok_encoding_free(TokEncoding* enc);
void tok_free(TokHandle* handle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ORT_SERVER_TOK_FFI_H
