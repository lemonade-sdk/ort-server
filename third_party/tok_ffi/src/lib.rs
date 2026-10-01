//! A C ABI over the HuggingFace `tokenizers` crate, for ort-server.
//!
//! `tok_encode` returns character offsets alongside the ids, which the zero-shot
//! router needs. Failures come back as null / `false`, never as a panic: C++
//! cannot catch a Rust panic, so one would abort the whole process.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use tokenizers::Tokenizer;

/// Opaque to C++: it only ever holds the pointer.
pub struct TokHandle {
    inner: Tokenizer,
}

/// One encoding. All three arrays have `len` elements and belong to the caller
/// until it passes them to `tok_encoding_free`.
#[repr(C)]
pub struct TokEncoding {
    ids: *mut u32,
    starts: *mut u32,
    ends: *mut u32,
    len: usize,
}

/// Load a tokenizer.json. Returns null on any failure, including a panic.
///
/// The file's `truncation` and `padding` settings are disabled, as
/// `transformers` does on every call unless asked: ort-server owns the token
/// budget, and padding would only feed the model filler.
///
/// # Safety
/// `json` must point to `json_len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn tok_new(json: *const u8, json_len: usize) -> *mut TokHandle {
    if json.is_null() {
        return ptr::null_mut();
    }
    catch_unwind(AssertUnwindSafe(|| {
        let bytes = unsafe { std::slice::from_raw_parts(json, json_len) };
        match Tokenizer::from_bytes(bytes) {
            Ok(mut t) => {
                // Cannot fail when clearing.
                let _ = t.with_truncation(None);
                t.with_padding(None);
                Box::into_raw(Box::new(TokHandle { inner: t }))
            }
            Err(_) => ptr::null_mut(),
        }
    }))
    .unwrap_or(ptr::null_mut())
}

/// Tokenize `text`, filling `out`. Returns false on any failure.
///
/// Safe to call concurrently on one handle: encoding takes `&Tokenizer`, which
/// is `Sync`.
///
/// # Safety
/// `handle` must come from `tok_new`, `text` must point to `text_len` readable
/// bytes, and `out` must point to writable `TokEncoding` storage.
#[no_mangle]
pub unsafe extern "C" fn tok_encode(
    handle: *const TokHandle,
    text: *const u8,
    text_len: usize,
    add_special_tokens: bool,
    out: *mut TokEncoding,
) -> bool {
    if handle.is_null() || text.is_null() || out.is_null() {
        return false;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let tok = unsafe { &(*handle).inner };
        let bytes = unsafe { std::slice::from_raw_parts(text, text_len) };
        let text = match std::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => return false,
        };

        // Not `encode`, which reports offsets in bytes: ort-server's label
        // ranges, like the Python reference, count characters.
        let enc = match tok.encode_char_offsets(text, add_special_tokens) {
            Ok(e) => e,
            Err(_) => return false,
        };

        let ids: Vec<u32> = enc.get_ids().to_vec();
        let offsets = enc.get_offsets();
        if offsets.len() != ids.len() {
            return false;
        }
        let starts: Vec<u32> = offsets.iter().map(|(s, _)| *s as u32).collect();
        let ends: Vec<u32> = offsets.iter().map(|(_, e)| *e as u32).collect();
        let len = ids.len();

        unsafe {
            *out = TokEncoding {
                ids: Box::into_raw(ids.into_boxed_slice()) as *mut u32,
                starts: Box::into_raw(starts.into_boxed_slice()) as *mut u32,
                ends: Box::into_raw(ends.into_boxed_slice()) as *mut u32,
                len,
            };
        }
        true
    }))
    .unwrap_or(false)
}

/// Release one `tok_encode` result. Safe to call twice; the second is a no-op.
///
/// # Safety
/// `enc` must be null, or a `TokEncoding` filled by a successful `tok_encode`.
#[no_mangle]
pub unsafe extern "C" fn tok_encoding_free(enc: *mut TokEncoding) {
    if enc.is_null() {
        return;
    }
    unsafe {
        let e = &mut *enc;
        if !e.ids.is_null() {
            // Each array was a Box<[T]>, so capacity == len.
            drop(Vec::from_raw_parts(e.ids, e.len, e.len));
            drop(Vec::from_raw_parts(e.starts, e.len, e.len));
            drop(Vec::from_raw_parts(e.ends, e.len, e.len));
        }
        e.ids = ptr::null_mut();
        e.starts = ptr::null_mut();
        e.ends = ptr::null_mut();
        e.len = 0;
    }
}

/// Release a tokenizer from `tok_new`.
///
/// # Safety
/// `handle` must be null, or a pointer returned by `tok_new` and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn tok_free(handle: *mut TokHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Byte-level tokenizer declaring truncation at 192 (make_tiny_router.py).
    const ROUTER: &[u8] = include_bytes!("../../../test/fixtures/tiny-router/tokenizer.json");
    // WordPiece tokenizer declaring Fixed padding to 64 (make_tiny_pad.py).
    const PADDED: &[u8] = include_bytes!("../../../test/fixtures/tiny-pad/tokenizer.json");

    fn empty() -> TokEncoding {
        TokEncoding {
            ids: ptr::null_mut(),
            starts: ptr::null_mut(),
            ends: ptr::null_mut(),
            len: 0,
        }
    }

    /// Encodes `text` with the tokenizer in `json`; returns each token's end offset.
    fn encode_ends(json: &[u8], text: &str) -> Vec<u32> {
        unsafe {
            let h = tok_new(json.as_ptr(), json.len());
            assert!(!h.is_null());
            let mut enc = empty();
            assert!(tok_encode(h, text.as_ptr(), text.len(), true, &mut enc));
            let ends = std::slice::from_raw_parts(enc.ends, enc.len).to_vec();
            tok_encoding_free(&mut enc);
            tok_encoding_free(&mut enc); // a second free must be a no-op
            tok_free(h);
            ends
        }
    }

    #[test]
    fn offsets_are_character_indexed() {
        // 9 chars, 12 bytes: a byte-offset build reports 12 for the last token.
        let ends = encode_ends(ROUTER, "emoji \u{1F680} x");
        assert_eq!(*ends.last().unwrap(), 9);
    }

    #[test]
    fn truncation_is_disabled() {
        let n = encode_ends(ROUTER, &"word ".repeat(1000)).len();
        assert!(
            n > 192,
            "tokenizer.json truncation still applied: {n} tokens"
        );
    }

    #[test]
    fn padding_is_disabled() {
        let n = encode_ends(PADDED, "hello world").len();
        assert!(n < 64, "tokenizer.json padding still applied: {n} tokens");
    }

    #[test]
    fn corrupt_json_returns_null_instead_of_panicking() {
        let bad = b"";
        assert!(unsafe { tok_new(bad.as_ptr(), bad.len()) }.is_null());
        let bad = b"{\"truncated\": ";
        assert!(unsafe { tok_new(bad.as_ptr(), bad.len()) }.is_null());
    }

    #[test]
    fn null_arguments_are_rejected() {
        assert!(unsafe { tok_new(ptr::null(), 0) }.is_null());
        let mut enc = empty();
        assert!(!unsafe { tok_encode(ptr::null(), b"x".as_ptr(), 1, true, &mut enc) });
        unsafe { tok_free(ptr::null_mut()) };
        unsafe { tok_encoding_free(ptr::null_mut()) };
    }
}
