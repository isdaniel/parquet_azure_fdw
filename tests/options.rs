use parquet_azure_fdw::azure::AuthMethod;
use parquet_azure_fdw::fdw::options::{
    parse_server_options_from_slice, parse_table_options_from_slice,
    parse_user_mapping_options_from_slice, validate_combo, UserMappingOptions,
};

#[test]
fn server_defaults_endpoint() {
    let o = parse_server_options_from_slice(&[
        ("account_name", "acct"),
        ("auth_method", "managed_identity"),
    ])
    .unwrap();
    assert_eq!(o.account_name, "acct");
    assert_eq!(o.endpoint, "blob.core.windows.net");
    assert_eq!(o.auth_method, AuthMethod::ManagedIdentity);
}

#[test]
fn server_custom_endpoint() {
    let o = parse_server_options_from_slice(&[
        ("account_name", "acct"),
        ("endpoint", "blob.core.usgovcloudapi.net"),
        ("auth_method", "aad_sp"),
    ])
    .unwrap();
    assert_eq!(o.endpoint, "blob.core.usgovcloudapi.net");
    assert_eq!(o.auth_method, AuthMethod::AadServicePrincipal);
}

#[test]
fn server_missing_account_name_errors() {
    let r = parse_server_options_from_slice(&[("auth_method", "managed_identity")]);
    assert!(r.is_err());
}

#[test]
fn server_missing_auth_method_errors() {
    let r = parse_server_options_from_slice(&[("account_name", "acct")]);
    assert!(r.is_err());
}

#[test]
fn server_rejects_account_key_auth() {
    let r = parse_server_options_from_slice(&[
        ("account_name", "acct"),
        ("auth_method", "account_key"),
    ]);
    let msg = format!("{}", r.unwrap_err());
    assert!(
        msg.contains("sas_url"),
        "remediation should mention sas_url: {msg}"
    );
}

#[test]
fn server_rejects_unknown_option() {
    let r = parse_server_options_from_slice(&[
        ("account_name", "acct"),
        ("auth_method", "managed_identity"),
        ("bogus", "x"),
    ]);
    assert!(r.is_err());
}

#[test]
fn enable_pushdown_defaults_true() {
    let s = parse_server_options_from_slice(&[
        ("account_name", "acct"),
        ("auth_method", "managed_identity"),
    ])
    .unwrap();
    assert!(s.enable_pushdown);
}

#[test]
fn enable_pushdown_off_parses() {
    let s = parse_server_options_from_slice(&[
        ("account_name", "acct"),
        ("auth_method", "managed_identity"),
        ("enable_pushdown", "false"),
    ])
    .unwrap();
    assert!(!s.enable_pushdown);
}

#[test]
fn user_mapping_accepts_sas_url() {
    let um = parse_user_mapping_options_from_slice(&[(
        "sas_url",
        "https://acct.blob.core.windows.net/c?sv=...",
    )])
    .unwrap();
    assert!(um.sas_url.is_some());
}

#[test]
fn user_mapping_rejects_account_key() {
    let r = parse_user_mapping_options_from_slice(&[("account_key", "abc=")]);
    let msg = format!("{}", r.unwrap_err());
    assert!(msg.contains("sas_url"));
}

#[test]
fn table_rejects_glob_with_dotdot() {
    let r = parse_table_options_from_slice(&[("container", "c"), ("filename", "../etc/passwd")]);
    assert!(r.is_err());
}

#[test]
fn table_rejects_absolute_filename() {
    let r =
        parse_table_options_from_slice(&[("container", "c"), ("filename", "/abs/path.parquet")]);
    assert!(r.is_err());
}

#[test]
fn table_accepts_simple_glob() {
    let o = parse_table_options_from_slice(&[
        ("container", "cont"),
        ("filename", "events/2024/*.parquet"),
    ])
    .unwrap();
    assert_eq!(o.container, "cont");
    assert_eq!(o.filename, "events/2024/*.parquet");
}

#[test]
fn table_missing_required_errors() {
    assert!(parse_table_options_from_slice(&[("container", "c")]).is_err());
    assert!(parse_table_options_from_slice(&[("filename", "f.parquet")]).is_err());
}

#[test]
fn combo_sas_url_requires_sas_option() {
    let s =
        parse_server_options_from_slice(&[("account_name", "acct"), ("auth_method", "sas_url")])
            .unwrap();
    let um = UserMappingOptions::default();
    assert!(validate_combo(&s, &um).is_err());

    let um_ok = parse_user_mapping_options_from_slice(&[(
        "sas_url",
        "https://acct.blob.core.windows.net/c?sig=x",
    )])
    .unwrap();
    validate_combo(&s, &um_ok).unwrap();
}

#[test]
fn combo_aad_sp_rejects_sas_option() {
    let s = parse_server_options_from_slice(&[("account_name", "acct"), ("auth_method", "aad_sp")])
        .unwrap();
    let um = parse_user_mapping_options_from_slice(&[(
        "sas_url",
        "https://acct.blob.core.windows.net/c?sig=x",
    )])
    .unwrap();
    assert!(validate_combo(&s, &um).is_err());
}

#[test]
fn combo_managed_identity_no_user_mapping() {
    let s = parse_server_options_from_slice(&[
        ("account_name", "acct"),
        ("auth_method", "managed_identity"),
    ])
    .unwrap();
    validate_combo(&s, &UserMappingOptions::default()).unwrap();
}

#[test]
fn server_rejects_endpoint_with_path() {
    // SSRF attempt — endpoint must be bare host[:port].
    let r = parse_server_options_from_slice(&[
        ("account_name", "acct"),
        ("endpoint", "blob.core.windows.net/evil"),
        ("auth_method", "managed_identity"),
    ]);
    assert!(r.is_err(), "endpoint with path must be rejected");
}

#[test]
fn server_rejects_endpoint_not_in_allowlist() {
    let r = parse_server_options_from_slice(&[
        ("account_name", "acct"),
        ("endpoint", "attacker.example.com"),
        ("auth_method", "managed_identity"),
    ]);
    let msg = format!("{}", r.unwrap_err());
    assert!(msg.contains("allowlist"), "{msg}");
}

#[test]
fn server_rejects_account_name_with_punctuation() {
    // SSRF attempt — would let attacker craft URL via account name.
    let r = parse_server_options_from_slice(&[
        ("account_name", "acct.evil.com/x?a"),
        ("endpoint", "blob.core.windows.net"),
        ("auth_method", "managed_identity"),
    ]);
    assert!(r.is_err());
}

#[test]
fn user_mapping_rejects_sas_to_imds() {
    let r = parse_user_mapping_options_from_slice(&[(
        "sas_url",
        "http://169.254.169.254/metadata/identity/oauth2/token?sig=x",
    )]);
    let msg = format!("{}", r.unwrap_err());
    assert!(
        msg.contains("private") || msg.contains("link-local") || msg.contains("https"),
        "{msg}"
    );
}

#[test]
fn user_mapping_rejects_sas_to_rfc1918() {
    let r = parse_user_mapping_options_from_slice(&[("sas_url", "https://10.0.0.5/c?sig=x")]);
    assert!(r.is_err());
}

#[test]
fn user_mapping_rejects_sas_to_lookalike_host() {
    // host must end in an azure suffix — naive substring check would let this through.
    let r = parse_user_mapping_options_from_slice(&[(
        "sas_url",
        "https://evil.com/blob.core.windows.net?sig=x",
    )]);
    assert!(r.is_err());
}

#[test]
fn user_mapping_accepts_azurite_loopback() {
    parse_user_mapping_options_from_slice(&[(
        "sas_url",
        "http://127.0.0.1:10000/devstoreaccount1/cont?sig=x",
    )])
    .unwrap();
}
