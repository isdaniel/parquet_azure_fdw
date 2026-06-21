use parquet_azure_fdw::azure::auth::{parse_auth_method, AuthMethod};
use parquet_azure_fdw::error::FdwError;

#[test]
fn parse_known_methods() {
    assert!(matches!(
        parse_auth_method("managed_identity").unwrap(),
        AuthMethod::ManagedIdentity
    ));
    assert!(matches!(
        parse_auth_method("aad_sp").unwrap(),
        AuthMethod::AadServicePrincipal
    ));
    assert!(matches!(
        parse_auth_method("sas_url").unwrap(),
        AuthMethod::SasUrl
    ));
}

#[test]
fn parse_unknown_method_is_error() {
    assert!(parse_auth_method("blah").is_err());
}

#[test]
fn account_key_is_explicitly_rejected_with_remediation() {
    let err = parse_auth_method("account_key").unwrap_err();
    let msg = err.to_string();
    assert!(
        matches!(err, FdwError::InvalidOption(_)),
        "expected InvalidOption, got {err:?}"
    );
    assert!(
        msg.contains("sas_url"),
        "error must point users to sas_url; got: {msg}"
    );
}
