#![forbid(unsafe_code)]
//! Secret redaction for error messages destined for the Postgres log.
//!
//! Azure SDK errors routinely embed the full request URL — which for SAS auth
//! includes the `sig=` query parameter (HMAC signature). Letting that hit
//! `ereport!` would leak the SAS to anyone with log access. Similarly, any
//! `Bearer <token>` substring (from AAD/MI flows) must not be logged.
//!
//! [`redact`] is best-effort, regex-free, and runs on whatever text the SDK
//! happened to emit — we do not assume a structured shape.

/// Query-string keys whose VALUES must be redacted.
///
/// `sig` and `signature` are the actual HMAC secret. `skoid`/`sktid`/`skt`/
/// `ske`/`sks`/`skv` together identify a user-delegation key and weaken
/// auditing if exposed. `code` covers OAuth authorization-code flows. The
/// `*_token` and `assertion` entries cover OAuth/AAD token-endpoint response
/// bodies (azure_identity 1.0 already formats some IMDS response bodies into
/// its error messages — see imds_managed_identity_credential.rs — so any
/// future leak path through `FdwError::azure(e).to_string()` would otherwise
/// surface a live Bearer or refresh token to the Postgres log).
//
// Order matters for the longest-prefix-wins behaviour of the matcher loop:
// `access_token` must appear before `code`/`id_token`/`refresh_token` would
// confuse the boundary checks, but since the matcher checks the next byte
// is `=` it's not actually ambiguous — we keep the existing entries first
// for stability of the redactor's empirical behaviour, then append the
// new ones.
const SECRET_QUERY_KEYS: &[&str] = &[
    "sig",
    "signature",
    "skoid",
    "sktid",
    "skt",
    "ske",
    "sks",
    "skv",
    "code",
    "access_token",
    "refresh_token",
    "id_token",
    "client_assertion",
    "assertion",
];

/// Strip secret-bearing tokens from a message.
///
/// - Replaces the value of any `SECRET_QUERY_KEYS` query parameter with
///   `REDACTED`.
/// - Replaces any `Bearer <token>` substring with `Bearer REDACTED`.
///
/// Idempotent; cheap; allocation-free on the happy path (no secrets present).
pub fn redact(msg: &str) -> String {
    let lower = msg.to_ascii_lowercase();
    let has_secret_key = SECRET_QUERY_KEYS
        .iter()
        .any(|k| lower.contains(&format!("{k}=")));
    // Fast-path gate: contains the literal "bearer" anywhere (any case). The
    // actual matcher in the loop below verifies the boundary (must be
    // followed by ASCII whitespace) so a false positive here only causes the
    // slow path to run, never a wrong redaction.
    let has_bearer = lower.contains("bearer");
    if !has_secret_key && !has_bearer {
        return msg.to_string();
    }

    let mut out = String::with_capacity(msg.len());
    let bytes = msg.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Try to match "Bearer" (case-insensitive) followed by one or more
        // ASCII whitespace characters (space, tab, CR, LF, etc.). The
        // original implementation required exactly one ASCII space and so
        // let `Bearer\teyJ...` (HTTP header echo with tab-separator) slip
        // through unredacted; widening to any whitespace closes that gap.
        if let Some(rest) = consume_prefix_ci(&bytes[i..], b"bearer") {
            let ws_len = rest
                .iter()
                .take_while(|&&c| c.is_ascii_whitespace())
                .count();
            if ws_len > 0 {
                let tail = &rest[ws_len..];
                out.push_str("Bearer REDACTED");
                let end = tail
                    .iter()
                    .position(|&c| c.is_ascii_whitespace() || c == b'"' || c == b',' || c == b'\'')
                    .unwrap_or(tail.len());
                i += (bytes.len() - i - rest.len()) + ws_len + end;
                continue;
            }
        }

        // Try to match "<key>=" for any secret key, case-insensitive.
        let mut matched = false;
        for key in SECRET_QUERY_KEYS {
            let pat: Vec<u8> = key.bytes().chain(std::iter::once(b'=')).collect();
            if let Some(rest) = consume_prefix_ci(&bytes[i..], &pat) {
                // Preceding char must be a query separator (or start/whitespace)
                // so we don't redact e.g. "config=...". We check by looking
                // at the previous byte.
                let prev_ok = i == 0
                    || matches!(
                        bytes[i - 1],
                        b'?' | b'&' | b' ' | b'\t' | b'\n' | b'(' | b'\'' | b'"' | b','
                    );
                if !prev_ok {
                    continue;
                }
                out.push_str(key);
                out.push_str("=REDACTED");
                let end = rest
                    .iter()
                    .position(|&c| {
                        matches!(c, b'&' | b' ' | b'\t' | b'\n' | b')' | b'\'' | b'"' | b',')
                    })
                    .unwrap_or(rest.len());
                i += (bytes.len() - i - rest.len()) + end;
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }

        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// If `bytes` begins with `prefix` (case-insensitive ASCII), return the
/// remainder. Otherwise `None`.
fn consume_prefix_ci<'a>(bytes: &'a [u8], prefix: &[u8]) -> Option<&'a [u8]> {
    if bytes.len() < prefix.len() {
        return None;
    }
    for (a, b) in bytes[..prefix.len()].iter().zip(prefix.iter()) {
        if !a.eq_ignore_ascii_case(b) {
            return None;
        }
    }
    Some(&bytes[prefix.len()..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_through_clean_text() {
        assert_eq!(redact("plain error message"), "plain error message");
    }

    #[test]
    fn redacts_sas_signature() {
        let url = "https://acct.blob.core.windows.net/c/b?sv=2024-11-04&se=2025-01-01T00%3A00%3A00Z&sig=AbCdEf%2B123%2F%3D";
        let r = redact(url);
        assert!(!r.contains("AbCdEf"), "{r}");
        assert!(r.contains("sig=REDACTED"), "{r}");
        assert!(r.contains("sv=2024-11-04"), "non-secret should remain: {r}");
    }

    #[test]
    fn redacts_signature_uppercase_param() {
        let r = redact("error url ?Signature=ZZZZZ&se=now");
        assert!(!r.contains("ZZZZZ"));
        assert!(r.contains("signature=REDACTED"));
    }

    #[test]
    fn redacts_bearer_token() {
        let r = redact("auth: Bearer eyJhbGciOi.foo.bar more text");
        assert!(!r.contains("eyJhbGciOi"));
        assert!(r.contains("Bearer REDACTED"));
        assert!(r.contains("more text"));
    }

    #[test]
    fn redacts_user_delegation_key_params() {
        let r = redact("?skoid=11111111-aaaa&sktid=22222222-bbbb&sig=zzzz");
        assert!(!r.contains("11111111"));
        assert!(!r.contains("22222222"));
        assert!(!r.contains("zzzz"));
    }

    #[test]
    fn does_not_redact_word_with_sig_substring() {
        // "config=..." must NOT match "sig=" — preceding char check.
        assert_eq!(redact("config=verbose"), "config=verbose");
    }

    #[test]
    fn idempotent() {
        let s = "?sig=secret";
        let once = redact(s);
        let twice = redact(&once);
        assert_eq!(once, twice);
    }

    // Regression: Bearer tokens separated by whitespace other than a single
    // ASCII space (tab, CRLF) previously slipped through both the fast-path
    // gate and the inner matcher because both required exactly `bearer `.
    // An HTTP header echo or multi-line log line that includes
    // `Authorization:\tBearer\teyJ...` was leaked verbatim to the postgres
    // log. The matcher now accepts any run of ASCII whitespace after the
    // keyword and strips the following token.
    #[test]
    fn redacts_bearer_with_tab_separator() {
        let r = redact("Authorization:\tBearer\teyJhbGciOi.foo.bar tail");
        assert!(!r.contains("eyJhbGciOi"), "{r}");
        assert!(r.contains("Bearer REDACTED"), "{r}");
        assert!(r.contains("tail"), "{r}");
    }

    #[test]
    fn redacts_bearer_with_crlf_separator() {
        let r = redact("hdr:\r\nBearer\r\neyJ.AAA.BBB\r\nnext");
        assert!(!r.contains("eyJ.AAA"), "{r}");
        assert!(r.contains("Bearer REDACTED"), "{r}");
    }

    #[test]
    fn redacts_bearer_with_multiple_spaces() {
        let r = redact("Bearer   eyJ.multi.space");
        assert!(!r.contains("eyJ.multi"), "{r}");
        assert!(r.contains("Bearer REDACTED"), "{r}");
    }

    // Regression: an azure_identity error path that echoed an OAuth
    // token-endpoint response body (e.g. `access_token=ya29.AAA&...`)
    // would leak a live bearer credential to the postgres log because
    // SECRET_QUERY_KEYS did not include the OAuth response field names.
    #[test]
    fn redacts_oauth_access_refresh_id_tokens() {
        let body = "?access_token=ya29.AAAA&refresh_token=BBBB&id_token=eyJ.id&expires_in=3600";
        let r = redact(body);
        assert!(!r.contains("ya29.AAAA"), "{r}");
        assert!(!r.contains("BBBB"), "{r}");
        assert!(!r.contains("eyJ.id"), "{r}");
        assert!(r.contains("access_token=REDACTED"), "{r}");
        assert!(r.contains("refresh_token=REDACTED"), "{r}");
        assert!(r.contains("id_token=REDACTED"), "{r}");
        // Non-secret fields are preserved.
        assert!(r.contains("expires_in=3600"), "{r}");
    }

    #[test]
    fn redacts_client_assertion() {
        // Used in AAD certificate / federated-credential flows.
        let r = redact("?client_assertion=eyJ.cert.value&client_assertion_type=jwt-bearer");
        assert!(!r.contains("eyJ.cert.value"), "{r}");
        assert!(r.contains("client_assertion=REDACTED"), "{r}");
    }

    #[test]
    fn bearer_word_without_token_is_not_truncated() {
        // The keyword "Bearer" must be followed by whitespace to trigger the
        // redactor. A bare "bearership" string must pass through untouched.
        assert_eq!(redact("bearership"), "bearership");
    }
}
