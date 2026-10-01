// C ABI over the HuggingFace `tokenizers` crate, implemented in src/lib.rs.
// TokEncoding must stay field-for-field in step with the #[repr(C)] struct there.
//
// Every successful tok_encode needs exactly one tok_encoding_free, and every
// non-null tok_new one tok_free. tok_encode may run concurrently on one handle.

#ifndef ORT_SERVER_TOK_FFI_H
#define ORT_SERVER_TOK_FFI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TokHandle TokHandle;  // opaque

// Offsets are CHARACTER indices into the input, never byte indices. Byte-level
// tokens that split one character all report that character's full span.
typedef struct {
    uint32_t* ids;     // token ids
    uint32_t* starts;  // per-token offset start
    uint32_t* ends;    // per-token offset end
    size_t len;        // element count of all three arrays
} TokEncoding;

// Returns NULL if the bytes are not a loadable tokenizer.json. The file's
// truncation and padding settings are disabled; the caller owns the budget.
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
