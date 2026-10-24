//! Print what `tok_encode` returns, so a Python harness can compare it with
//! the HuggingFace reference. Driven by `test/parity_tokenizer.py`.
//!
//! This calls the FFI entry points, not `tokenizers::Tokenizer` directly. That
//! is the whole point: the crate is upstream's problem, and this layer --
//! `encode_char_offsets`, the u32 narrowing, the offsets/ids length check -- is
//! ours. An example that bypassed it would pass while this server was broken.
//!
//!     cargo run --release --example dump -- <tokenizer.json> [--no-special]
//!
//! Protocol, chosen so that an input may hold ANY byte, newlines included:
//!
//!   stdin   per record: a line with the decimal byte count, then exactly that
//!           many bytes of UTF-8, then one newline.
//!   stdout  per record: one JSON object, `{"ids":[..],"starts":[..],"ends":[..]}`
//!           or `{"error":"..."}`. Every value is an integer, so this file
//!           formats the JSON itself and needs no serde dependency.

use std::io::{self, BufRead, Read, Write};
use std::mem::MaybeUninit;

use tok_ffi::{tok_encode, tok_encoding_free, tok_free, tok_new, TokEncoding};

fn die(msg: &str) -> ! {
    eprintln!("dump: {msg}");
    std::process::exit(2);
}

fn write_array(out: &mut impl Write, name: &str, v: &[u32], last: bool) -> io::Result<()> {
    write!(out, "\"{name}\":[")?;
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            write!(out, ",")?;
        }
        write!(out, "{x}")?;
    }
    write!(out, "]{}", if last { "" } else { "," })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = match args.first() {
        Some(p) if !p.starts_with("--") => p.clone(),
        _ => die("usage: dump <tokenizer.json> [--no-special]"),
    };
    // Default ON, matching ort-server's own encode() call. The flag exists
    // because the inserted markers are the part most likely to differ.
    let add_special = !args.iter().any(|a| a == "--no-special");

    let blob = std::fs::read(&path).unwrap_or_else(|e| die(&format!("cannot read {path}: {e}")));
    let handle = unsafe { tok_new(blob.as_ptr(), blob.len()) };
    if handle.is_null() {
        die(&format!("not a loadable tokenizer.json: {path}"));
    }

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    let mut header = String::new();
    loop {
        header.clear();
        match reader.read_line(&mut header) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => die(&format!("stdin: {e}")),
        }
        let trimmed = header.trim();
        if trimmed.is_empty() {
            continue;
        }
        let n: usize = trimmed
            .parse()
            .unwrap_or_else(|_| die(&format!("bad length header: {trimmed:?}")));

        let mut buf = vec![0u8; n];
        if let Err(e) = reader.read_exact(&mut buf) {
            die(&format!("stdin: short record: {e}"));
        }
        let mut nl = [0u8; 1];
        if let Err(e) = reader.read_exact(&mut nl) {
            die(&format!("stdin: missing record terminator: {e}"));
        }

        // zeroed(), not a literal: the struct is only written on success, and
        // tok_encoding_free reads the null ids pointer as "nothing to free".
        let mut enc = MaybeUninit::<TokEncoding>::zeroed();
        let ok = unsafe {
            tok_encode(handle, buf.as_ptr(), buf.len(), add_special, enc.as_mut_ptr())
        };
        let mut enc = unsafe { enc.assume_init() };

        let r = if ok {
            let (ids, starts, ends) = unsafe {
                (
                    std::slice::from_raw_parts(enc.ids, enc.len),
                    std::slice::from_raw_parts(enc.starts, enc.len),
                    std::slice::from_raw_parts(enc.ends, enc.len),
                )
            };
            (|| -> io::Result<()> {
                write!(out, "{{")?;
                write_array(&mut out, "ids", ids, false)?;
                write_array(&mut out, "starts", starts, false)?;
                write_array(&mut out, "ends", ends, true)?;
                writeln!(out, "}}")
            })()
        } else {
            writeln!(out, "{{\"error\":\"tok_encode failed\"}}")
        };
        if let Err(e) = r {
            die(&format!("stdout: {e}"));
        }
        unsafe { tok_encoding_free(&mut enc) };

        // Flush per record: the harness reads line by line and would otherwise
        // wait on a buffer that only fills at the end of a long run.
        if let Err(e) = out.flush() {
            die(&format!("stdout: {e}"));
        }
    }

    unsafe { tok_free(handle) };
}
