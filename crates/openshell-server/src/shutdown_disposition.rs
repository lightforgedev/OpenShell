// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a graceful gateway shutdown does with running sandboxes.
//!
//! The default, [`ShutdownDisposition::StopRunningSandboxes`], is the
//! historical behaviour: the gateway sends a driver stop to every persisted
//! sandbox with running intent before it exits. An embedded host that
//! restarts its gateway can instead select
//! [`ShutdownDisposition::RetainRunningSandboxes`], so the next gateway can
//! re-adopt the live sandboxes. Retain skips only that stop sweep; listener
//! shutdown, the managed driver process and every other cleanup still run.
//!
//! The disposition is separate from the driver's `gateway_manages_lifecycle`
//! capability, which also gates startup reconciliation and must stay on for
//! re-adoption to work.
//!
//! Two sources select it:
//!
//! - a static value (`--shutdown-disposition`,
//!   `OPENSHELL_GATEWAY_SHUTDOWN_DISPOSITION`), and
//! - a disposition file (`--shutdown-disposition-file`,
//!   `OPENSHELL_GATEWAY_SHUTDOWN_DISPOSITION_FILE`) read when the shutdown
//!   signal arrives, so a host can pick the mode per shutdown. When a file is
//!   configured it is the only source.
//!
//! Anything other than an exact, known value selects the stop sweep: an
//! unset value, an unknown value, a missing, unreadable or oversized file.

use std::io::Read;
use std::path::{Path, PathBuf};

/// Largest disposition file the gateway reads. Valid values are short tokens.
const MAX_DISPOSITION_FILE_BYTES: u64 = 64;

/// What graceful shutdown does with sandboxes that have running intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShutdownDisposition {
    /// Stop every persisted sandbox with running intent (the default).
    #[default]
    StopRunningSandboxes,
    /// Leave running sandboxes in place for the next gateway to re-adopt.
    RetainRunningSandboxes,
}

impl ShutdownDisposition {
    pub const STOP: &'static str = "stop-running-sandboxes";
    pub const RETAIN: &'static str = "retain-running-sandboxes";

    /// Parse an exact disposition token. Surrounding whitespace is ignored;
    /// anything else, including case variants, is unknown.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim() {
            Self::STOP => Some(Self::StopRunningSandboxes),
            Self::RETAIN => Some(Self::RetainRunningSandboxes),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StopRunningSandboxes => Self::STOP,
            Self::RetainRunningSandboxes => Self::RETAIN,
        }
    }
}

/// Where a resolved disposition came from, for the shutdown receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispositionSource {
    /// Nothing was configured.
    Default,
    /// The static flag or environment value.
    Static,
    /// The disposition file.
    File,
}

impl DispositionSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Static => "static",
            Self::File => "file",
        }
    }
}

/// Why a configured source fell back to the stop sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispositionFallback {
    Unknown,
    Missing,
    Unreadable,
    Oversized,
}

impl DispositionFallback {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Missing => "missing",
            Self::Unreadable => "unreadable",
            Self::Oversized => "oversized",
        }
    }
}

/// A disposition with its provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedDisposition {
    pub mode: ShutdownDisposition,
    pub source: DispositionSource,
    /// Set when a configured source could not be used and the stop sweep was
    /// selected instead.
    pub fallback: Option<DispositionFallback>,
}

/// Configured disposition sources.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShutdownDispositionConfig {
    pub value: Option<String>,
    pub file: Option<PathBuf>,
}

impl ShutdownDispositionConfig {
    /// Resolve the disposition now. Call when the shutdown signal arrives so
    /// a disposition file reflects the host's latest choice.
    pub fn resolve(&self) -> ResolvedDisposition {
        if let Some(path) = &self.file {
            return match read_disposition_file(path) {
                Ok(raw) => from_token(&raw, DispositionSource::File),
                Err(fallback) => stop(DispositionSource::File, Some(fallback)),
            };
        }
        self.value.as_deref().map_or_else(
            || stop(DispositionSource::Default, None),
            |raw| from_token(raw, DispositionSource::Static),
        )
    }
}

const fn stop(
    source: DispositionSource,
    fallback: Option<DispositionFallback>,
) -> ResolvedDisposition {
    ResolvedDisposition {
        mode: ShutdownDisposition::StopRunningSandboxes,
        source,
        fallback,
    }
}

fn from_token(raw: &str, source: DispositionSource) -> ResolvedDisposition {
    ShutdownDisposition::parse(raw).map_or_else(
        || stop(source, Some(DispositionFallback::Unknown)),
        |mode| ResolvedDisposition {
            mode,
            source,
            fallback: None,
        },
    )
}

fn read_disposition_file(path: &Path) -> Result<String, DispositionFallback> {
    let file = std::fs::File::open(path).map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            DispositionFallback::Missing
        } else {
            DispositionFallback::Unreadable
        }
    })?;
    let mut bytes = Vec::new();
    file.take(MAX_DISPOSITION_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| DispositionFallback::Unreadable)?;
    if bytes.len() as u64 > MAX_DISPOSITION_FILE_BYTES {
        return Err(DispositionFallback::Oversized);
    }
    String::from_utf8(bytes).map_err(|_| DispositionFallback::Unknown)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(value: Option<&str>, file: Option<PathBuf>) -> ShutdownDispositionConfig {
        ShutdownDispositionConfig {
            value: value.map(str::to_owned),
            file,
        }
    }

    #[test]
    fn missing_disposition_defaults_to_stop_sweep() {
        assert_eq!(
            config(None, None).resolve(),
            ResolvedDisposition {
                mode: ShutdownDisposition::StopRunningSandboxes,
                source: DispositionSource::Default,
                fallback: None,
            }
        );
    }

    #[test]
    fn known_static_values_parse_exactly() {
        assert_eq!(
            config(Some("retain-running-sandboxes"), None)
                .resolve()
                .mode,
            ShutdownDisposition::RetainRunningSandboxes
        );
        assert_eq!(
            config(Some(" stop-running-sandboxes\n"), None)
                .resolve()
                .mode,
            ShutdownDisposition::StopRunningSandboxes
        );
    }

    #[test]
    fn unknown_static_value_defaults_to_stop_sweep() {
        for raw in [
            "",
            "retain",
            "RETAIN-RUNNING-SANDBOXES",
            "skip-cleanup",
            "retain-running-sandboxes,extra",
        ] {
            assert_eq!(
                config(Some(raw), None).resolve(),
                ResolvedDisposition {
                    mode: ShutdownDisposition::StopRunningSandboxes,
                    source: DispositionSource::Static,
                    fallback: Some(DispositionFallback::Unknown),
                },
                "{raw:?} must select the stop sweep"
            );
        }
    }

    #[test]
    fn disposition_file_is_read_at_resolve_time() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("disposition");
        let cfg = config(Some("retain-running-sandboxes"), Some(path.clone()));

        let missing = cfg.resolve();
        assert_eq!(
            missing,
            ResolvedDisposition {
                mode: ShutdownDisposition::StopRunningSandboxes,
                source: DispositionSource::File,
                fallback: Some(DispositionFallback::Missing),
            },
            "a configured but missing file must not fall back to the static value"
        );

        std::fs::write(&path, "retain-running-sandboxes\n").unwrap();
        assert_eq!(
            cfg.resolve(),
            ResolvedDisposition {
                mode: ShutdownDisposition::RetainRunningSandboxes,
                source: DispositionSource::File,
                fallback: None,
            }
        );

        std::fs::write(&path, "stop-running-sandboxes").unwrap();
        assert_eq!(
            cfg.resolve().mode,
            ShutdownDisposition::StopRunningSandboxes
        );
    }

    #[test]
    fn unknown_or_oversized_disposition_file_defaults_to_stop_sweep() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("disposition");
        let cfg = config(None, Some(path.clone()));

        std::fs::write(&path, "keep-everything").unwrap();
        assert_eq!(cfg.resolve().fallback, Some(DispositionFallback::Unknown));

        std::fs::write(&path, [0xff, 0xfe]).unwrap();
        assert_eq!(cfg.resolve().fallback, Some(DispositionFallback::Unknown));

        let mut oversized = "retain-running-sandboxes".to_string();
        oversized.push_str(&" ".repeat(64));
        std::fs::write(&path, oversized).unwrap();
        let resolved = cfg.resolve();
        assert_eq!(resolved.mode, ShutdownDisposition::StopRunningSandboxes);
        assert_eq!(resolved.fallback, Some(DispositionFallback::Oversized));

        assert_eq!(
            config(None, Some(dir.path().to_path_buf())).resolve().mode,
            ShutdownDisposition::StopRunningSandboxes,
            "a directory is not a disposition file"
        );
    }
}
