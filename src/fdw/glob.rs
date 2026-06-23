#![forbid(unsafe_code)]
#![allow(dead_code)] // consumed by SP-2 Task 2 (PlanSource) onward
//! Glob → regex translation for blob path patterns.
//!
//! Supports `*` (matches any chars except `/`) and `?` (matches any single
//! char except `/`). Rejects `**` (recursive — out of scope), `..`
//! (defense-in-depth alongside the SSRF validator in `options.rs`), empty
//! patterns, and absolute paths (leading `/`).
//!
//! The translation:
//!   - longest no-wildcard prefix is passed to Azure's list_with_prefix_etags
//!     to bound the result set up front.
//!   - the full pattern becomes an anchored regex; client-side filtering
//!     narrows the LIST result to the exact match set.

use crate::error::{FdwError, FdwResult};
use regex::Regex;

pub struct GlobPattern {
    pub prefix: String,
    pub regex: Regex,
}

pub fn parse_glob(pattern: &str) -> FdwResult<GlobPattern> {
    if pattern.is_empty() {
        return Err(FdwError::InvalidOption("filename is empty".into()));
    }
    if pattern.starts_with('/') {
        return Err(FdwError::InvalidOption(
            "filename must be a blob name within the container, not an absolute path".into(),
        ));
    }
    if pattern.contains("..") {
        return Err(FdwError::InvalidOption(
            "filename must not contain '..' path segments".into(),
        ));
    }
    if pattern.contains("**") {
        return Err(FdwError::InvalidOption(
            "recursive glob '**' is not supported in v1".into(),
        ));
    }

    // Longest no-wildcard prefix.
    let first_wild = pattern.find(['*', '?']).unwrap_or(pattern.len());
    let prefix = pattern[..first_wild].to_string();

    // Build the regex by walking the pattern. Escape every literal char,
    // translate '*' → '[^/]*' and '?' → '[^/]'.
    let mut re_src = String::with_capacity(pattern.len() * 2 + 2);
    re_src.push('^');
    for c in pattern.chars() {
        match c {
            '*' => re_src.push_str("[^/]*"),
            '?' => re_src.push_str("[^/]"),
            // Escape only real regex metacharacters. The old "escape everything
            // not ASCII-alphanumeric" rule produced `\ä`-style escapes for
            // Unicode chars, which the regex crate rejects — breaking any glob
            // containing non-ASCII alphanumerics.
            '.' | '+' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\' => {
                re_src.push('\\');
                re_src.push(c);
            }
            _ => re_src.push(c),
        }
    }
    re_src.push('$');

    let regex = Regex::new(&re_src)
        .map_err(|e| FdwError::InvalidOption(format!("invalid glob '{pattern}': {e}")))?;

    Ok(GlobPattern { prefix, regex })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trailing_star_matches_extension() {
        let g = parse_glob("data/*.parquet").unwrap();
        assert_eq!(g.prefix, "data/");
        assert!(g.regex.is_match("data/2026.parquet"));
        assert!(g.regex.is_match("data/users.parquet"));
        assert!(!g.regex.is_match("data/sub/2026.parquet"));
        assert!(!g.regex.is_match("other/2026.parquet"));
    }

    #[test]
    fn mid_segment_star_works() {
        let g = parse_glob("logs/*/access.log").unwrap();
        assert_eq!(g.prefix, "logs/");
        assert!(g.regex.is_match("logs/2026/access.log"));
        assert!(!g.regex.is_match("logs/2026/sub/access.log"));
        assert!(!g.regex.is_match("logs/access.log"));
    }

    #[test]
    fn unicode_chars_compile_and_match() {
        // Regression: non-ASCII alphanumerics must not be backslash-escaped
        // (the regex crate rejects `\ä`-style escapes). The glob must compile
        // and match literally.
        let g = parse_glob("数据/*.parquet").unwrap();
        assert!(g.regex.is_match("数据/2026.parquet"));
        assert!(!g.regex.is_match("other/2026.parquet"));
        // A real metacharacter is still escaped (treated literally, not regex).
        let g2 = parse_glob("a.b/*.parquet").unwrap();
        assert!(g2.regex.is_match("a.b/x.parquet"));
        assert!(!g2.regex.is_match("axb/x.parquet"));
    }

    #[test]
    fn question_mark_matches_single_char() {
        let g = parse_glob("v?/data.parquet").unwrap();
        assert_eq!(g.prefix, "v");
        assert!(g.regex.is_match("v1/data.parquet"));
        assert!(g.regex.is_match("vA/data.parquet"));
        assert!(!g.regex.is_match("v10/data.parquet"));
        assert!(!g.regex.is_match("v/data.parquet"));
    }

    #[test]
    fn literal_no_wildcards_returns_anchored_match() {
        let g = parse_glob("data.parquet").unwrap();
        assert_eq!(g.prefix, "data.parquet");
        assert!(g.regex.is_match("data.parquet"));
        assert!(!g.regex.is_match("data.parquet.bak"));
        assert!(!g.regex.is_match("xdata.parquet"));
    }

    #[test]
    fn rejects_recursive_glob() {
        assert!(parse_glob("**/file.parquet").is_err());
    }

    #[test]
    fn rejects_traversal() {
        assert!(parse_glob("../etc/passwd").is_err());
        assert!(parse_glob("data/../secrets").is_err());
    }

    #[test]
    fn rejects_absolute_path() {
        assert!(parse_glob("/etc/passwd").is_err());
    }

    #[test]
    fn rejects_empty() {
        assert!(parse_glob("").is_err());
    }

    #[test]
    fn literal_dot_does_not_act_as_regex_metacharacter() {
        // 'data.parquet' must NOT match 'dataXparquet' even though `.` is
        // a regex metachar.
        let g = parse_glob("data.parquet").unwrap();
        assert!(!g.regex.is_match("dataXparquet"));
    }
}
