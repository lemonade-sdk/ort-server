//! A C ABI over the HuggingFace `tokenizers` crate, for ort-server.
//!
//! Four functions, and two properties worth stating up front:
//!
//!   * `tok_encode` returns the character offsets alongside the ids. The
//!     tokenizer computes them anyway, and the zero-shot router needs them to
//!     work out which tokens belong to which label.
//!   * Failures come back as null / `false`, never as a panic across the FFI
//!     boundary. C++ cannot catch a Rust panic, so an `unwrap()` on a corrupt
//!     tokenizer.json would abort the whole process.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use tokenizers::Tokenizer;

/// Opaque to C++: it only ever holds the pointer.
pub struct TokHandle {
    inner: Tokenizer,
}

/// One encoding. All three arrays have `len` elements.
///
/// Ownership passes to the caller, which must hand each one back via
/// `tok_encoding_free` exactly once -- the same contract as malloc/free.
#[repr(C)]
pub struct TokEncoding {
    ids: *mut u32,
    starts: *mut u32,
    ends: *mut u32,
    len: usize,
}

/// Load a tokenizer.json. Returns null on any failure, including a panic.
///
/// Any `truncation` block in the file is DISABLED here, deliberately, so that
/// the token budget has exactly one owner: ort-server's `max_length`, which
/// defaults to this same value and can be overridden by the manifest. Left
/// enabled, the crate cuts the sequence before ort-server ever sees it, which
/// silently caps a manifest that asks for a larger window and makes the
/// resulting error message name a budget that was never applied.
///
/// This is not a behaviour change at the default: the crate truncates before
/// post-processing and ort-server cuts after it, but this file's template adds
/// only a PREFIX special token and truncates right, so the two produce
/// identical ids and offsets. Verified on plain, router-shaped and
/// unicode-heavy inputs, and pinned by `truncation_is_left_to_the_caller`.
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
                // Infallible when clearing: `with_truncation(None)` cannot fail.
                let _ = t.with_truncation(None);
                Box::into_raw(Box::new(TokHandle { inner: t }))
            }
            Err(_) => ptr::null_mut(),
        }
    }))
    .unwrap_or(ptr::null_mut())
}

/// Tokenize `text`, filling `out`. Returns false on any failure.
///
/// The result is NEVER truncated here (see `tok_new`) -- the caller owns the
/// token budget and must apply it to all three arrays together.
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

        // `encode_char_offsets`, not `encode`: the plain form reports offsets in
        // BYTES of the input, while the Python reference this server is
        // validated against (and the label ranges ort-server computes) work in
        // CHARACTERS. For "emoji <rocket> x" the last token is (7,9) here and
        // (7,12) from `encode` -- the same tokens, incompatible units.
        //
        // Not `encode_fast` either: the fast variant skips computing offsets,
        // which is the one thing this function exists for.
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
                // Release ownership; C++ returns it via tok_encoding_free.
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

    /// The offsets must be character-indexed, or every label range ort-server
    /// computes lands in the wrong place on any non-ASCII input. A byte-offset
    /// build reports (7, 12) for the trailing token here.
    ///
    /// The fixture is the byte-level tokenizer from
    /// `test/fixtures/make_tiny_router.py`; regenerate it with that script.
    #[test]
    fn offsets_are_character_indexed() {
        let json = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/tokenizer.json"
        ));
        let Ok(json) = json else {
            eprintln!("skipped: no tests/data/tokenizer.json fixture");
            return;
        };
        let h = unsafe { tok_new(json.as_ptr(), json.len()) };
        assert!(!h.is_null());

        let text = "emoji \u{1F680} x"; // 9 chars, 12 bytes
        let mut enc = TokEncoding {
            ids: ptr::null_mut(),
            starts: ptr::null_mut(),
            ends: ptr::null_mut(),
            len: 0,
        };
        assert!(unsafe { tok_encode(h, text.as_ptr(), text.len(), true, &mut enc) });
        let ends = unsafe { std::slice::from_raw_parts(enc.ends, enc.len) };
        assert_eq!(*ends.last().unwrap(), 9, "offsets are in bytes, not chars");

        unsafe {
            tok_encoding_free(&mut enc);
            tok_encoding_free(&mut enc); // double free must be a no-op
            tok_free(h);
        }
    }

    /// tok_new must clear tokenizer.json's own truncation, leaving ort-server
    /// the single owner of the token budget. The fixture declares
    /// `max_length: 192`; if the crate were still applying it, a manifest
    /// asking for a larger window would be silently capped here and the
    /// resulting error would name a budget that was never applied.
    #[test]
    fn truncation_is_left_to_the_caller() {
        let json = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/tokenizer.json"
        ));
        let Ok(json) = json else {
            eprintln!("skipped: no tests/data/tokenizer.json fixture");
            return;
        };
        let h = unsafe { tok_new(json.as_ptr(), json.len()) };
        assert!(!h.is_null());

        let text = "word ".repeat(1000); // far past the fixture's 192
        let mut enc = TokEncoding {
            ids: ptr::null_mut(),
            starts: ptr::null_mut(),
            ends: ptr::null_mut(),
            len: 0,
        };
        assert!(unsafe { tok_encode(h, text.as_ptr(), text.len(), true, &mut enc) });
        assert!(
            enc.len > 192,
            "tokenizer.json truncation still applied: got {} tokens",
            enc.len
        );

        unsafe {
            tok_encoding_free(&mut enc);
            tok_free(h);
        }
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
        let mut enc = TokEncoding {
            ids: ptr::null_mut(),
            starts: ptr::null_mut(),
            ends: ptr::null_mut(),
            len: 0,
        };
        assert!(!unsafe { tok_encode(ptr::null(), b"x".as_ptr(), 1, true, &mut enc) });
        unsafe { tok_free(ptr::null_mut()) };
        unsafe { tok_encoding_free(ptr::null_mut()) };
    }
}
