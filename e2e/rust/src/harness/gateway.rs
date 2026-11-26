// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gateway process controls for Docker-backed e2e tests.
//!
//! The shell wrapper still prepares the expensive shared setup: binaries,
//! PKI, state directories, Docker network, and the first gateway launch. This
//! helper owns restart mechanics inside Rust tests by reading the wrapper's
//! exported process metadata.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

/// A gateway process owned by the current Docker e2e wrapper run.
pub struct ManagedGateway {
    bin: PathBuf,
    args_file: PathBuf,
    log: PathBuf,
    pid_file: PathBuf,
}

impl ManagedGateway {
    /// Build controls from explicit wrapper-provided process metadata.
    pub fn new(
        bin: impl Into<PathBuf>,
        args_file: impl Into<PathBuf>,
        log: impl Into<PathBuf>,
        pid_file: impl Into<PathBuf>,
    ) -> Self {
        Self {
            bin: bin.into(),
            args_file: args_file.into(),
            log: log.into(),
            pid_file: pid_file.into(),
        }
    }

    /// Load managed gateway controls from the environment.
    ///
    /// Returns `Ok(None)` when the current e2e run does not own the gateway,
    /// such as `OPENSHELL_GATEWAY_ENDPOINT=http://...` existing-endpoint mode.
    pub fn from_env() -> Result<Option<Self>, String> {
        let Some(bin) = std::env::var_os("OPENSHELL_E2E_GATEWAY_BIN") else {
            return Ok(None);
        };

        Ok(Some(Self {
            bin: PathBuf::from(bin),
            args_file: env_path("OPENSHELL_E2E_GATEWAY_ARGS_FILE")?,
            log: env_path("OPENSHELL_E2E_GATEWAY_LOG")?,
            pid_file: env_path("OPENSHELL_E2E_GATEWAY_PID_FILE")?,
        }))
    }

    /// Start the gateway if it is not already running.
    pub fn start(&self) -> Result<(), String> {
        self.start_with_args(&[])
    }

    /// Start the gateway with `extra` appended to the wrapper's arguments, if
    /// it is not already running. A later [`Self::start`] drops them again.
    pub fn start_with_args(&self, extra: &[&str]) -> Result<(), String> {
        if let Some(pid) = self.current_pid()? {
            if process_running(pid)? {
                return Ok(());
            }
            let _ = fs::remove_file(&self.pid_file);
        }

        let mut args = self.gateway_args()?;
        args.extend(extra.iter().map(|arg| (*arg).to_string()));
        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log)
            .map_err(|err| format!("open gateway log '{}': {err}", self.log.display()))?;
        writeln!(
            log,
            "\n=== starting openshell-gateway from Rust e2e harness ==="
        )
        .map_err(|err| format!("write gateway log marker: {err}"))?;
        let stderr = log
            .try_clone()
            .map_err(|err| format!("clone gateway log handle: {err}"))?;

        let child = Command::new(&self.bin)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(|err| format!("start openshell-gateway '{}': {err}", self.bin.display()))?;
        let pid = child.id();
        fs::write(&self.pid_file, format!("{pid}\n")).map_err(|err| {
            format!(
                "write gateway pid file '{}': {err}",
                self.pid_file.display()
            )
        })?;

        Ok(())
    }

    /// Stop the gateway if it is running.
    pub fn stop(&self) -> Result<(), String> {
        let Some(pid) = self.current_pid()? else {
            return Ok(());
        };
        if !process_running(pid)? {
            let _ = fs::remove_file(&self.pid_file);
            return Ok(());
        }

        send_signal(pid, None)?;
        for _ in 0..60 {
            if !process_running(pid)? {
                let _ = fs::remove_file(&self.pid_file);
                return Ok(());
            }
            thread::sleep(Duration::from_secs(1));
        }

        send_signal(pid, Some("-9"))?;
        let _ = fs::remove_file(&self.pid_file);
        Ok(())
    }

    /// Send `SIGTERM` and wait for the gateway's graceful shutdown to finish.
    ///
    /// Unlike [`Self::stop`], this never escalates to `SIGKILL`: a gateway
    /// that does not exit within `timeout` is an error, so callers can rely
    /// on graceful shutdown cleanup having run. Only the PID recorded by this
    /// e2e run is signalled.
    pub fn terminate(&self, timeout: Duration) -> Result<(), String> {
        let Some(pid) = self.current_pid()? else {
            return Err("gateway is not running".to_string());
        };
        if !process_running(pid)? {
            let _ = fs::remove_file(&self.pid_file);
            return Err(format!("gateway process {pid} already exited"));
        }
        send_signal(pid, None)?;
        let deadline = std::time::Instant::now() + timeout;
        while std::time::Instant::now() < deadline {
            if !process_running(pid)? {
                let _ = fs::remove_file(&self.pid_file);
                return Ok(());
            }
            thread::sleep(Duration::from_millis(250));
        }
        Err(format!(
            "gateway process {pid} did not exit within {}s of SIGTERM",
            timeout.as_secs()
        ))
    }

    /// The gateway log shared by every start in this e2e run.
    pub fn log_path(&self) -> &std::path::Path {
        &self.log
    }

    /// Kill the gateway with `SIGKILL`, skipping graceful shutdown.
    ///
    /// This models an embedded gateway whose host process dies: running
    /// sandbox containers and their supervisor sidecars are left in place for
    /// the next gateway start to re-adopt. Only the PID recorded by this e2e
    /// run is signalled.
    pub fn kill(&self) -> Result<(), String> {
        let Some(pid) = self.current_pid()? else {
            return Ok(());
        };
        if process_running(pid)? {
            send_signal(pid, Some("-9"))?;
            for _ in 0..60 {
                if !process_running(pid)? {
                    break;
                }
                thread::sleep(Duration::from_millis(500));
            }
            if process_running(pid)? {
                return Err(format!("gateway process {pid} survived SIGKILL"));
            }
        }
        let _ = fs::remove_file(&self.pid_file);
        Ok(())
    }

    fn current_pid(&self) -> Result<Option<u32>, String> {
        let Ok(raw) = fs::read_to_string(&self.pid_file) else {
            return Ok(None);
        };
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        trimmed.parse::<u32>().map(Some).map_err(|err| {
            format!(
                "parse gateway pid file '{}': {err}",
                self.pid_file.display()
            )
        })
    }

    fn gateway_args(&self) -> Result<Vec<String>, String> {
        let raw = fs::read(&self.args_file)
            .map_err(|err| format!("read gateway args '{}': {err}", self.args_file.display()))?;
        raw.split(|byte| *byte == 0)
            .filter(|arg| !arg.is_empty())
            .map(|arg| {
                String::from_utf8(arg.to_vec()).map_err(|err| {
                    format!(
                        "gateway args file '{}' is not UTF-8: {err}",
                        self.args_file.display()
                    )
                })
            })
            .collect()
    }
}

impl Drop for ManagedGateway {
    fn drop(&mut self) {
        let _ = self.start();
    }
}

fn env_path(name: &str) -> Result<PathBuf, String> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| format!("{name} must be set when OPENSHELL_E2E_GATEWAY_BIN is set"))
}

fn process_running(pid: u32) -> Result<bool, String> {
    if !signal_command(["-0", &pid.to_string()])? {
        return Ok(false);
    }

    let output = Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "stat="])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("run ps for pid {pid}: {err}"))?;
    if !output.status.success() {
        return Ok(false);
    }

    let stat = String::from_utf8_lossy(&output.stdout);
    Ok(!stat.trim_start().starts_with('Z'))
}

fn send_signal(pid: u32, signal: Option<&str>) -> Result<(), String> {
    let mut args = Vec::new();
    if let Some(signal) = signal {
        args.push(signal);
    }
    let pid_string = pid.to_string();
    args.push(&pid_string);
    if signal_command(args)? {
        Ok(())
    } else {
        Err(format!("failed to signal gateway process {pid}"))
    }
}

fn signal_command<'a>(args: impl IntoIterator<Item = &'a str>) -> Result<bool, String> {
    Command::new("kill")
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .map_err(|err| format!("run kill: {err}"))
}
