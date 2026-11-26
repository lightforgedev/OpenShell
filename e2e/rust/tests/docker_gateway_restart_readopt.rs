// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "e2e")]

//! E2E coverage for re-adopting a live Docker sandbox across repeated
//! ungraceful gateway restarts.
//!
//! An embedded gateway can die without running its graceful shutdown sweep,
//! so the sandbox container keeps running. The next gateway start removes the
//! stale supervisor sidecar and spawns a replacement. That replacement must
//! re-attach to the same, still-running sandbox boundary: the sandbox id and
//! container stay the same, the workload is not restarted, the sidecar becomes
//! healthy, `sandbox exec` works and the driver-owned volumes are retained.
//!
//! The test targets the gateway started by `e2e/with-docker-gateway.sh`.
//! Existing-endpoint runs do not own the gateway process and skip it.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use openshell_e2e::harness::cli::{
    run_cli, wait_for_healthy, wait_for_sandbox_exec_contains, wait_for_sandbox_phase,
};
use openshell_e2e::harness::gateway::ManagedGateway;
use openshell_e2e::harness::output::strip_ansi;
use openshell_e2e::harness::sandbox::SandboxGuard;
use tokio::time::sleep;

const READY_MARKER: &str = "gateway-readopt-ready";
const STATE_FILE: &str = "/sandbox/gateway-readopt-state";
const RESTARTS: usize = 3;
const MANAGED_BY_LABEL: &str = "openshell.ai/managed-by=openshell";
const SANDBOX_ID_LABEL: &str = "openshell.ai/sandbox-id";
const SANDBOX_NAME_LABEL: &str = "openshell.ai/sandbox-name";
const SANDBOX_NAMESPACE_LABEL: &str = "openshell.ai/sandbox-namespace";
const ROLE_LABEL: &str = "openshell.ai/isolation-role";

fn docker(args: &[&str]) -> Result<String, String> {
    let output = Command::new("docker")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("failed to run docker {}: {err}", args.join(" ")))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !output.status.success() {
        return Err(format!(
            "docker {} failed (exit {:?}):\n{stdout}{}",
            args.join(" "),
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(stdout)
}

fn lines(output: &str) -> Vec<String> {
    output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn containers_with_role(
    namespace: &str,
    sandbox_name: &str,
    role: &str,
) -> Result<Vec<String>, String> {
    let namespace_filter = format!("label={SANDBOX_NAMESPACE_LABEL}={namespace}");
    let name_filter = format!("label={SANDBOX_NAME_LABEL}={sandbox_name}");
    let role_filter = format!("label={ROLE_LABEL}={role}");
    let managed_filter = format!("label={MANAGED_BY_LABEL}");
    docker(&[
        "ps",
        "-aq",
        "--no-trunc",
        "--filter",
        &managed_filter,
        "--filter",
        &namespace_filter,
        "--filter",
        &name_filter,
        "--filter",
        &role_filter,
    ])
    .map(|output| lines(&output))
}

fn single_container(namespace: &str, sandbox_name: &str, role: &str) -> Result<String, String> {
    let ids = containers_with_role(namespace, sandbox_name, role)?;
    match ids.as_slice() {
        [id] => Ok(id.clone()),
        _ => Err(format!(
            "expected one {role} container for sandbox '{sandbox_name}', found {ids:?}"
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ContainerState {
    running: bool,
    started_at: String,
    restart_count: String,
    health: String,
}

fn container_state(id: &str) -> Result<ContainerState, String> {
    let output = docker(&[
        "inspect",
        "-f",
        "{{.State.Running}}|{{.State.StartedAt}}|{{.RestartCount}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
        id,
    ])?;
    let fields = output.split('|').collect::<Vec<_>>();
    let [running, started_at, restart_count, health] = fields.as_slice() else {
        return Err(format!(
            "unexpected docker inspect output for {id}: {output}"
        ));
    };
    Ok(ContainerState {
        running: *running == "true",
        started_at: (*started_at).to_string(),
        restart_count: (*restart_count).to_string(),
        health: (*health).to_string(),
    })
}

fn sandbox_id_label(container_id: &str) -> Result<String, String> {
    let format = format!("{{{{index .Config.Labels \"{SANDBOX_ID_LABEL}\"}}}}");
    let id = docker(&["inspect", "-f", &format, container_id])?;
    if id.is_empty() {
        return Err(format!("container {container_id} has no sandbox id label"));
    }
    Ok(id)
}

fn sandbox_volumes(sandbox_id: &str) -> Result<Vec<String>, String> {
    let filter = format!("label={SANDBOX_ID_LABEL}={sandbox_id}");
    let mut names = lines(&docker(&["volume", "ls", "-q", "--filter", &filter])?);
    names.sort();
    Ok(names)
}

/// Probe the supervisor's readiness socket the same way its Docker
/// healthcheck does (the image has no shell).
fn supervisor_health_socket_present(supervisor_id: &str) -> bool {
    docker(&[
        "exec",
        supervisor_id,
        "/openshell-supervisor",
        "health",
        "--socket",
        "/run/openshell/health.sock",
    ])
    .is_ok()
}

/// Raw `docker logs` bytes (stdout then stderr) of a container.
fn docker_logs_raw(container_id: &str) -> Result<String, String> {
    let output = Command::new("docker")
        .args(["logs", container_id])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("failed to run docker logs {container_id}: {err}"))?;
    Ok(format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

/// The boundary's handover log line for `generation`, which must carry its
/// structured fields (the shorthand layer used to drop every non-message
/// field, so the cause of a rejected attach never reached `docker logs`).
fn handover_log_line(container_id: &str, generation: usize) -> Result<String, String> {
    let logs = docker_logs_raw(container_id)?;
    let expected = [
        "code=\"CK_supervisor_handover_accepted\"".to_string(),
        format!("previous_supervisor_generation=Some({})", generation - 1),
        format!(" supervisor_generation=Some({generation})"),
    ];
    logs.lines()
        .find(|line| expected.iter().all(|field| line.contains(field.as_str())))
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            format!(
                "no handover log line with fields {expected:?} in docker logs of {container_id}"
            )
        })
}

/// Distinct boundary log lines from the sandbox container, for failure output.
/// Repeated lines (for example a retried exchange failure) are collapsed.
fn boundary_log_summary(container_id: &str) -> String {
    let output = Command::new("docker")
        .args(["logs", container_id])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    let Ok(output) = output else {
        return "<docker logs failed>".to_string();
    };
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let mut seen = std::collections::BTreeSet::new();
    let mut lines = Vec::new();
    for line in text.lines() {
        if line.contains("Landlock") {
            continue;
        }
        // Drop the timestamp so repeated events collapse.
        let key = line
            .split_once(' ')
            .map_or(line, |(_, rest)| rest)
            .to_string();
        if seen.insert(key) {
            lines.push(line.to_string());
        }
    }
    lines.join("\n")
}

async fn wait_for_healthy_supervisor(
    namespace: &str,
    sandbox_name: &str,
    timeout: Duration,
) -> Result<String, String> {
    let deadline = Instant::now() + timeout;
    let mut last = String::new();
    while Instant::now() < deadline {
        match containers_with_role(namespace, sandbox_name, "supervisor") {
            Ok(ids) if ids.len() == 1 => match container_state(&ids[0]) {
                Ok(state) if state.running && state.health == "healthy" => {
                    return Ok(ids[0].clone());
                }
                Ok(state) => last = format!("supervisor {}: {state:?}", ids[0]),
                Err(err) => last = err,
            },
            Ok(ids) => last = format!("supervisor containers: {ids:?}"),
            Err(err) => last = err,
        }
        sleep(Duration::from_millis(500)).await;
    }
    Err(format!(
        "sandbox '{sandbox_name}' supervisor did not become healthy within {}s. Last: {last}",
        timeout.as_secs()
    ))
}

async fn sandbox_field(sandbox_name: &str, field: &str) -> String {
    let (output, _) = run_cli(&["sandbox", "get", sandbox_name, "--output", "json"]).await;
    let clean = strip_ansi(&output);
    serde_json::from_str::<serde_json::Value>(&clean)
        .ok()
        .and_then(|value| value.get(field).and_then(|p| p.as_str()).map(str::to_owned))
        .unwrap_or(clean)
}

async fn sandbox_phase(sandbox_name: &str) -> String {
    sandbox_field(sandbox_name, "phase").await
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn docker_gateway_restarts_readopt_live_sandbox() {
    let Some(gateway) = ManagedGateway::from_env().expect("load managed e2e gateway metadata")
    else {
        eprintln!("Skipping gateway re-adopt test: e2e gateway is not managed by this test run");
        return;
    };
    let Some(namespace) = std::env::var("OPENSHELL_E2E_SANDBOX_NAMESPACE")
        .ok()
        .filter(|value| !value.trim().is_empty())
    else {
        eprintln!("Skipping gateway re-adopt test: Docker e2e sandbox namespace is unavailable");
        return;
    };

    wait_for_healthy(Duration::from_secs(30))
        .await
        .expect("gateway should start healthy");

    let script = format!(
        "echo before-restart > {STATE_FILE}; echo {READY_MARKER}; while true; do sleep 1; done"
    );
    let mut sandbox = SandboxGuard::create_keep(&["sh", "-lc", &script], READY_MARKER)
        .await
        .expect("create long-running sandbox");
    wait_for_sandbox_phase(&sandbox.name, "Ready", Duration::from_secs(120))
        .await
        .expect("sandbox should be Ready before restarts");

    let container_id =
        single_container(&namespace, &sandbox.name, "sandbox").expect("sandbox container");
    let sandbox_id = sandbox_id_label(&container_id).expect("sandbox id label");
    assert_eq!(
        sandbox_field(&sandbox.name, "id").await,
        sandbox_id,
        "gateway sandbox id should match the container's sandbox id label"
    );
    let initial_state = container_state(&container_id).expect("sandbox container state");
    assert!(initial_state.running, "sandbox container should be running");
    let initial_volumes = sandbox_volumes(&sandbox_id).expect("list sandbox volumes");
    assert!(
        initial_volumes
            .iter()
            .any(|name| name.starts_with("openshell-supervisor-"))
            && initial_volumes
                .iter()
                .any(|name| name.starts_with("openshell-channel-")),
        "sandbox should own a channel and a supervisor volume, found {initial_volumes:?}"
    );
    let mut supervisor_id =
        wait_for_healthy_supervisor(&namespace, &sandbox.name, Duration::from_secs(60))
            .await
            .expect("initial supervisor should be healthy");
    // Baseline: exec must work before any restart, so a post-restart exec
    // failure is attributable to re-adoption.
    wait_for_sandbox_exec_contains(
        &sandbox.name,
        &["cat", STATE_FILE],
        "before-restart",
        Duration::from_secs(120),
    )
    .await
    .expect("sandbox exec should work before any gateway restart");

    for restart in 1..=RESTARTS {
        eprintln!("gateway re-adopt: restart {restart}/{RESTARTS}");
        gateway.kill().expect("kill e2e gateway");

        let state = container_state(&container_id).expect("sandbox state after gateway kill");
        assert!(
            state.running,
            "restart {restart}: ungraceful gateway exit must leave the sandbox running"
        );

        gateway.start().expect("restart e2e gateway");
        wait_for_healthy(Duration::from_secs(120))
            .await
            .unwrap_or_else(|err| panic!("restart {restart}: gateway not healthy: {err}"));

        let replacement =
            wait_for_healthy_supervisor(&namespace, &sandbox.name, Duration::from_secs(120))
                .await
                .unwrap_or_else(|err| {
                    panic!(
                        "restart {restart}: {err}\nboundary log:\n{}",
                        boundary_log_summary(&container_id)
                    )
                });
        assert_ne!(
            replacement, supervisor_id,
            "restart {restart}: gateway startup should replace the stale supervisor sidecar"
        );
        assert!(
            supervisor_health_socket_present(&replacement),
            "restart {restart}: replacement supervisor has no /run/openshell/health.sock"
        );
        supervisor_id = replacement;

        if let Err(err) =
            wait_for_sandbox_phase(&sandbox.name, "Ready", Duration::from_secs(120)).await
        {
            panic!(
                "restart {restart}: sandbox did not return to Ready (phase {}): {err}\nboundary log:\n{}",
                sandbox_phase(&sandbox.name).await,
                boundary_log_summary(&container_id)
            );
        }
        wait_for_sandbox_exec_contains(
            &sandbox.name,
            &["cat", STATE_FILE],
            "before-restart",
            Duration::from_secs(120),
        )
        .await
        .unwrap_or_else(|err| {
            panic!(
                "restart {restart}: sandbox exec failed: {err}\nboundary log:\n{}",
                boundary_log_summary(&container_id)
            )
        });

        assert_eq!(
            single_container(&namespace, &sandbox.name, "sandbox").expect("sandbox container"),
            container_id,
            "restart {restart}: the same sandbox container must be retained"
        );
        assert_eq!(
            sandbox_field(&sandbox.name, "id").await,
            sandbox_id,
            "restart {restart}: the gateway must re-adopt the same sandbox id"
        );
        let state = container_state(&container_id).expect("sandbox state after re-adopt");
        assert!(
            state.running,
            "restart {restart}: sandbox should be running"
        );
        assert_eq!(
            (state.started_at.as_str(), state.restart_count.as_str()),
            (
                initial_state.started_at.as_str(),
                initial_state.restart_count.as_str()
            ),
            "restart {restart}: the sandbox workload must not be restarted"
        );
        assert_eq!(
            sandbox_volumes(&sandbox_id).expect("list sandbox volumes"),
            initial_volumes,
            "restart {restart}: sandbox volumes must be retained"
        );
        // Generation 1 is the sidecar started at create; restart N hands
        // over to generation N + 1.
        let handover = handover_log_line(&container_id, restart + 1)
            .unwrap_or_else(|err| panic!("restart {restart}: {err}"));
        eprintln!("gateway re-adopt: restart {restart} boundary log: {handover}");
    }

    sandbox.cleanup().await;

    // Delete owns volume cleanup. Remove leftovers only for this test's
    // sandbox id so a failed delete does not leak volumes between runs.
    let leftovers = sandbox_volumes(&sandbox_id).unwrap_or_default();
    for volume in &leftovers {
        let _ = docker(&["volume", "rm", "-f", volume]);
    }
    assert!(
        leftovers.is_empty(),
        "sandbox delete should remove its volumes, found {leftovers:?}"
    );
}
