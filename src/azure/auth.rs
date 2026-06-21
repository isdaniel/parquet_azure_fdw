#![forbid(unsafe_code)]
//! Auth method parsing and credential construction for Azure Blob Storage.
//!
//! Three auth methods are supported, selected by the `auth_method` FDW option:
//!
//! - `managed_identity` — system-assigned MI via IMDS / App Service.
//! - `aad_sp` — AAD service principal; reads `AZURE_TENANT_ID`,
//!   `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` from env.
//! - `sas_url` — full container SAS URL passed in `sas_url` option.
//!
//! Shared-key (account_key) auth is not supported because
//! `azure_storage_blob` 1.0 accepts only `TokenCredential`. Users with an
//! account key should generate a container SAS client-side and use `sas_url`.

use crate::error::{FdwError, FdwResult};
use std::sync::Arc;

use azure_core::credentials::{Secret, TokenCredential};
use azure_identity::{ClientSecretCredential, ManagedIdentityCredential};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthMethod {
    ManagedIdentity,
    AadServicePrincipal,
    SasUrl,
}

pub fn parse_auth_method(s: &str) -> FdwResult<AuthMethod> {
    match s {
        "managed_identity" => Ok(AuthMethod::ManagedIdentity),
        "aad_sp" => Ok(AuthMethod::AadServicePrincipal),
        "sas_url" => Ok(AuthMethod::SasUrl),
        "account_key" => Err(FdwError::InvalidOption(
            "auth_method='account_key' is not supported by azure_storage_blob 1.0; \
             generate a container SAS client-side and use auth_method='sas_url'"
                .into(),
        )),
        other => Err(FdwError::InvalidOption(format!("auth_method='{other}'"))),
    }
}

/// Materialized credential ready to be plumbed into a blob client.
pub enum Credential {
    Token(Arc<dyn TokenCredential>),
    SasUrl { container_url: String },
}

/// Build the runtime credential object selected by `method`.
///
/// `sas_url` is read from the matching `Option<&str>` arg; AAD
/// service-principal pulls `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`,
/// `AZURE_CLIENT_SECRET` from process env (matches Azure SDK convention).
pub fn build_credential(
    method: &AuthMethod,
    _account_name: &str,
    sas_url: Option<&str>,
) -> FdwResult<Credential> {
    match method {
        AuthMethod::ManagedIdentity => {
            let c = ManagedIdentityCredential::new(None).map_err(FdwError::azure)?;
            Ok(Credential::Token(c as Arc<dyn TokenCredential>))
        }
        AuthMethod::AadServicePrincipal => {
            let tenant_id = std::env::var("AZURE_TENANT_ID")
                .map_err(|_| FdwError::MissingOption("AZURE_TENANT_ID"))?;
            let client_id = std::env::var("AZURE_CLIENT_ID")
                .map_err(|_| FdwError::MissingOption("AZURE_CLIENT_ID"))?;
            let client_secret = std::env::var("AZURE_CLIENT_SECRET")
                .map_err(|_| FdwError::MissingOption("AZURE_CLIENT_SECRET"))?;
            let c = ClientSecretCredential::new(
                &tenant_id,
                client_id,
                Secret::new(client_secret),
                None,
            )
            .map_err(FdwError::azure)?;
            Ok(Credential::Token(c as Arc<dyn TokenCredential>))
        }
        AuthMethod::SasUrl => {
            let url = sas_url.ok_or(FdwError::MissingOption("sas_url"))?;
            Ok(Credential::SasUrl {
                container_url: url.to_string(),
            })
        }
    }
}
