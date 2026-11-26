// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared relay primitives for authorized explicit-proxy egress.

use super::{EgressDecision, L7RouteSnapshot, emit_l7_tunnel_close_after_stale_state};
use crate::l7::relay::L7EvalContext;
use crate::opa::{
    CredentialDependency, NetworkAction, OpaEngine, PolicyGenerationGuard, TunnelPolicyEngine,
};
use miette::{IntoDiagnostic, Result};
use openshell_core::activity::ActivitySender;
use openshell_core::endpoint_status::EndpointObservationSender;
use openshell_core::proto::ProviderProfileCredential;
use openshell_core::secrets::SecretResolver;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncWrite};

type DynamicCredentials = Arc<std::sync::RwLock<HashMap<String, ProviderProfileCredential>>>;

enum PreparedHttpPolicy {
    Inspect {
        configs: Vec<crate::l7::L7EndpointConfig>,
        evaluator: Box<TunnelPolicyEngine>,
    },
    Passthrough {
        generation_guard: PolicyGenerationGuard,
    },
}

/// Everything an HTTP relay needs after authorization is complete.
///
/// The relay deliberately owns a generation-pinned policy primitive instead
/// of retaining access to the mutable OPA engine. Policy reloads therefore
/// fail closed through the guard or tunnel evaluator already attached here.
pub(super) struct RelayContext<'a> {
    request: &'a L7EvalContext,
    policy: PreparedHttpPolicy,
    middleware_engine: &'a OpaEngine,
}

/// Non-blocking observation channels attached to an authorized HTTP relay.
pub(super) struct RelaySignals {
    /// Receives general sandbox network activity.
    pub(super) activity: Option<ActivitySender>,
    /// Receives terminal tool server results for endpoint status reporting.
    pub(super) endpoint_observation: Option<EndpointObservationSender>,
}

/// Build the request-processing context shared by CONNECT and forward HTTP.
pub(super) fn http_context(
    decision: &EgressDecision,
    provider_credentials: Option<openshell_core::provider_credentials::ProviderCredentialState>,
    secret_resolver: Option<Arc<SecretResolver>>,
    dynamic_credentials: Option<DynamicCredentials>,
    agent_proposals: openshell_core::proposals::AgentProposals,
    workspace: String,
    signals: RelaySignals,
) -> L7EvalContext {
    // Provider-backed credentials must be acquired from the live state for
    // each request after middleware/token-grant awaits. Keep only the legacy
    // resolver fallback when no live provider state exists.
    let secret_resolver = provider_credentials
        .is_none()
        .then_some(secret_resolver)
        .flatten();
    let policy_name = match &decision.action {
        NetworkAction::Allow { matched_policy } => matched_policy.clone().unwrap_or_default(),
        NetworkAction::Deny { .. } => String::new(),
    };

    L7EvalContext {
        host: decision.intent.destination.host.clone(),
        port: decision.intent.destination.port,
        request_default_port: None,
        policy_name,
        binary_path: decision
            .binary
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_default(),
        ancestors: decision
            .ancestors
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
        cmdline_paths: decision
            .cmdline_paths
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
        secret_resolver,
        provider_credentials,
        provider_credential_revision: None,
        body_classifier: None,
        activity_tx: signals.activity,
        dynamic_credentials: dynamic_credentials.clone(),
        token_grant_resolver: dynamic_credentials
            .as_ref()
            .map(|_| crate::l7::token_grant_injection::default_resolver()),
        agent_proposals,
        workspace,
        endpoint_observation_tx: signals.endpoint_observation,
    }
}

/// Pin a generation for a relay or the forward HTTP single-request path.
///
/// `dependency` must be [`CredentialDependency::Dependent`] for any path that
/// can resolve or inject provider credentials, so a credential-only
/// activation still closes it.
pub(super) fn pin_policy_generation(
    opa_engine: &OpaEngine,
    expected_generation: u64,
    dependency: CredentialDependency,
) -> Result<PolicyGenerationGuard> {
    opa_engine.generation_guard_for(expected_generation, dependency)
}

/// Clone an L7 evaluator for a relay or the forward HTTP single-request path.
pub(super) fn pin_l7_evaluator(
    opa_engine: &OpaEngine,
    expected_generation: u64,
) -> Result<TunnelPolicyEngine> {
    opa_engine.clone_engine_for_tunnel(expected_generation)
}

pub(super) fn validate_route_generation(
    route: Option<&L7RouteSnapshot>,
    expected_generation: u64,
) -> Result<()> {
    if let Some(route) = route
        && route.l7_policy_generation != expected_generation
    {
        return Err(miette::miette!(
            "policy changed before CONNECT route hydration \
             [l4_generation:{} l7_generation:{}]",
            expected_generation,
            route.l7_policy_generation,
        ));
    }
    Ok(())
}

/// Prepare a generation-pinned HTTP relay at the adapter boundary.
///
/// A stale generation preserves the established CONNECT behavior: emit the
/// policy-change close event and let the adapter close the live tunnel without
/// attempting to write an HTTP response into it.
pub(super) fn prepare_http_relay<'a>(
    route: Option<&L7RouteSnapshot>,
    opa_engine: &'a OpaEngine,
    decision: &EgressDecision,
    request: &'a L7EvalContext,
) -> Option<RelayContext<'a>> {
    if let Err(error) = validate_route_generation(route, decision.policy_generation) {
        emit_l7_tunnel_close_after_stale_state(
            &decision.intent.destination.host,
            decision.intent.destination.port,
            error,
        );
        return None;
    }

    let policy = if let Some(route) = route.filter(|route| !route.configs.is_empty()) {
        let evaluator = match pin_l7_evaluator(opa_engine, decision.policy_generation) {
            Ok(evaluator) => evaluator,
            Err(error) => {
                emit_l7_tunnel_close_after_stale_state(
                    &decision.intent.destination.host,
                    decision.intent.destination.port,
                    error,
                );
                return None;
            }
        };
        let configs = route
            .configs
            .iter()
            .map(|snapshot| snapshot.config.clone())
            .collect();
        PreparedHttpPolicy::Inspect {
            configs,
            evaluator: Box::new(evaluator),
        }
    } else {
        // Passthrough HTTP still rewrites credential placeholders.
        let generation_guard = match pin_policy_generation(
            opa_engine,
            decision.policy_generation,
            CredentialDependency::Dependent,
        ) {
            Ok(guard) => guard,
            Err(error) => {
                emit_l7_tunnel_close_after_stale_state(
                    &decision.intent.destination.host,
                    decision.intent.destination.port,
                    error,
                );
                return None;
            }
        };
        PreparedHttpPolicy::Passthrough { generation_guard }
    };

    Some(RelayContext {
        request,
        policy,
        middleware_engine: opa_engine,
    })
}

/// Pin the generation used by a raw relay so policy activation or quarantine
/// closes streams that otherwise have no request boundary at which to notice
/// a stale decision.
///
/// A raw relay copies bytes and never resolves or injects credentials; the
/// caller passes [`CredentialDependency::Dependent`] only when the endpoint is
/// credential-bound (an explicitly allowed uninspected credential route).
pub(super) fn prepare_raw_relay(
    route: Option<&L7RouteSnapshot>,
    opa_engine: &OpaEngine,
    decision: &EgressDecision,
    dependency: CredentialDependency,
) -> Option<PolicyGenerationGuard> {
    if let Err(error) = validate_route_generation(route, decision.policy_generation) {
        emit_l7_tunnel_close_after_stale_state(
            &decision.intent.destination.host,
            decision.intent.destination.port,
            error,
        );
        return None;
    }

    match pin_policy_generation(opa_engine, decision.policy_generation, dependency) {
        Ok(guard) => Some(guard),
        Err(error) => {
            emit_l7_tunnel_close_after_stale_state(
                &decision.intent.destination.host,
                decision.intent.destination.port,
                error,
            );
            None
        }
    }
}

/// Relay an HTTP/1 stream using an already-authorized, generation-pinned context.
///
/// CONNECT plaintext and TLS-terminated streams both enter through this
/// function. Forward HTTP will provide a buffered first request to the same
/// boundary in the next migration step.
pub(super) async fn relay_http_stream<C, U>(
    client: &mut C,
    upstream: &mut U,
    context: RelayContext<'_>,
) -> Result<()>
where
    C: AsyncRead + AsyncWrite + Unpin + Send,
    U: AsyncRead + AsyncWrite + Unpin + Send,
{
    match context.policy {
        PreparedHttpPolicy::Inspect { configs, evaluator } if configs.len() == 1 => {
            let generation_guard = evaluator.generation_guard().clone();
            tokio::select! {
                result = crate::l7::relay::relay_with_inspection(
                    &configs[0],
                    *evaluator,
                    client,
                    upstream,
                    context.request,
                ) => result,
                () = generation_guard.wait_until_stale() => {
                    emit_stale_relay_close(context.request, &generation_guard);
                    Ok(())
                }
            }
        }
        PreparedHttpPolicy::Inspect { configs, evaluator } => {
            let generation_guard = evaluator.generation_guard().clone();
            tokio::select! {
                result = crate::l7::relay::relay_with_route_selection(
                    &configs,
                    *evaluator,
                    client,
                    upstream,
                    context.request,
                ) => result,
                () = generation_guard.wait_until_stale() => {
                    emit_stale_relay_close(context.request, &generation_guard);
                    Ok(())
                }
            }
        }
        PreparedHttpPolicy::Passthrough { generation_guard } => {
            tokio::select! {
                result = crate::l7::relay::relay_passthrough_with_credentials(
                    client,
                    upstream,
                    context.request,
                    &generation_guard,
                    Some(context.middleware_engine),
                ) => result,
                () = generation_guard.wait_until_stale() => {
                    emit_stale_relay_close(context.request, &generation_guard);
                    Ok(())
                }
            }
        }
    }
}

/// Relay a policy-authorized raw TCP stream.
pub(super) async fn relay_tcp<C, U>(
    client: &mut C,
    upstream: &mut U,
    generation_guard: &PolicyGenerationGuard,
    request: &L7EvalContext,
) -> Result<()>
where
    C: AsyncRead + AsyncWrite + Unpin,
    U: AsyncRead + AsyncWrite + Unpin,
{
    tokio::select! {
        result = tokio::io::copy_bidirectional(client, upstream) => {
            result.into_diagnostic()?;
        }
        () = generation_guard.wait_until_stale() => {
            emit_stale_relay_close(request, generation_guard);
        }
    }
    Ok(())
}

fn emit_stale_relay_close(request: &L7EvalContext, guard: &PolicyGenerationGuard) {
    // `wait_until_stale` also returns when the generation channel closes; that
    // is an engine teardown, reported with the policy counters it observed.
    let error = guard.ensure_current().err().unwrap_or_else(|| {
        miette::miette!(
            "policy generation channel closed [captured_generation:{} current_generation:{}]",
            guard.captured_generation(),
            guard.current_generation(),
        )
    });
    emit_l7_tunnel_close_after_stale_state(&request.host, request.port, error);
}

#[cfg(test)]
mod tests {
    use super::super::{EgressIntent, EndpointDecision, ProcessIdentityEvidence};
    use super::*;

    const POLICY_REGO: &str = include_str!("../../data/sandbox-policy.rego");
    const EMPTY_POLICY_DATA: &str = "network_policies: {}\n";

    fn decision(policy_generation: u64) -> EgressDecision {
        EgressDecision {
            intent: EgressIntent::connect("example.com".to_string(), 80),
            action: NetworkAction::Allow {
                matched_policy: Some("test".to_string()),
            },
            policy_generation,
            identity: ProcessIdentityEvidence::Available,
            endpoint: EndpointDecision::default(),
            binary: None,
            binary_pid: None,
            ancestors: vec![],
            cmdline_paths: vec![],
        }
    }

    fn request_context() -> L7EvalContext {
        L7EvalContext {
            host: "example.com".to_string(),
            port: 80,
            request_default_port: Some(80),
            policy_name: "test".to_string(),
            binary_path: String::new(),
            ancestors: vec![],
            cmdline_paths: vec![],
            secret_resolver: None,
            provider_credentials: None,
            provider_credential_revision: None,
            body_classifier: None,
            activity_tx: None,
            dynamic_credentials: None,
            token_grant_resolver: None,
            agent_proposals: openshell_core::proposals::AgentProposals::default(),
            workspace: String::new(),
            endpoint_observation_tx: None,
        }
    }

    #[test]
    fn relay_without_route_pins_l4_decision_generation() {
        let engine = OpaEngine::from_strings(POLICY_REGO, EMPTY_POLICY_DATA).unwrap();
        let decision = decision(engine.current_generation());
        let request = request_context();

        let context = prepare_http_relay(None, &engine, &decision, &request)
            .expect("current L4 generation should prepare a relay");
        let PreparedHttpPolicy::Passthrough { generation_guard } = context.policy else {
            panic!("route-less relay should use a generation guard");
        };

        assert_eq!(
            generation_guard.captured_generation(),
            decision.policy_generation
        );
    }

    #[test]
    fn empty_hydrated_route_cannot_replace_stale_l4_generation() {
        let engine = OpaEngine::from_strings(POLICY_REGO, EMPTY_POLICY_DATA).unwrap();
        let decision = decision(u64::MAX);
        let route = L7RouteSnapshot {
            configs: vec![],
            l7_policy_generation: engine.current_generation(),
        };
        let request = request_context();

        assert!(
            prepare_http_relay(Some(&route), &engine, &decision, &request).is_none(),
            "a current L7 lookup must not freshen a stale L4 allow"
        );
    }

    #[test]
    fn inspected_route_cannot_replace_stale_l4_generation() {
        let engine = OpaEngine::from_strings(POLICY_REGO, EMPTY_POLICY_DATA).unwrap();
        let decision = decision(u64::MAX);
        let route = L7RouteSnapshot {
            configs: vec![super::super::L7ConfigSnapshot {
                config: crate::l7::L7EndpointConfig {
                    protocol: crate::l7::L7Protocol::Rest,
                    endpoint_id: String::new(),
                    policy_hash: String::new(),
                    path: "/**".to_string(),
                    tls: crate::l7::TlsMode::Auto,
                    enforcement: crate::l7::EnforcementMode::Enforce,
                    graphql_max_body_bytes: crate::l7::graphql::DEFAULT_MAX_BODY_BYTES,
                    json_rpc_max_body_bytes: crate::l7::jsonrpc::DEFAULT_MAX_BODY_BYTES,
                    mcp_strict_tool_names: true,
                    mcp_versions: Vec::new(),
                    allow_encoded_slash: false,
                    websocket_credential_rewrite: false,
                    request_body_credential_rewrite: false,
                    allow_uninspected_credentials: false,
                    provider_credentialed: false,
                    websocket_graphql_policy: false,
                    credential_signing: crate::l7::CredentialSigning::None,
                    signing_service: String::new(),
                    signing_region: String::new(),
                },
            }],
            l7_policy_generation: engine.current_generation(),
        };
        let request = request_context();

        assert!(
            prepare_http_relay(Some(&route), &engine, &decision, &request).is_none(),
            "an inspected route must use the generation that authorized CONNECT"
        );
    }

    #[test]
    fn raw_route_cannot_replace_stale_l4_generation() {
        let engine = OpaEngine::from_strings(POLICY_REGO, EMPTY_POLICY_DATA).unwrap();
        let decision = decision(u64::MAX);
        let route = L7RouteSnapshot {
            configs: vec![],
            l7_policy_generation: engine.current_generation(),
        };

        assert!(
            prepare_raw_relay(
                Some(&route),
                &engine,
                &decision,
                CredentialDependency::Independent
            )
            .is_none(),
            "a raw relay must not freshen a stale L4 allow"
        );
    }

    #[test]
    fn stale_generation_fails_before_relay_context_is_created() {
        let engine = OpaEngine::from_strings(POLICY_REGO, EMPTY_POLICY_DATA).unwrap();
        let decision = decision(engine.current_generation());
        let request = request_context();
        engine.reload(POLICY_REGO, EMPTY_POLICY_DATA).unwrap();

        assert!(
            prepare_http_relay(None, &engine, &decision, &request).is_none(),
            "policy reload must prevent a stale relay from starting"
        );
    }

    #[tokio::test]
    async fn raw_relay_closes_immediately_when_fail_closed_generation_is_published() {
        let engine = Arc::new(OpaEngine::from_strings(POLICY_REGO, EMPTY_POLICY_DATA).unwrap());
        let guard = engine
            .generation_guard(engine.current_generation())
            .unwrap();
        let request = request_context();
        let (_client_peer, mut proxy_client) = tokio::io::duplex(64);
        let (_upstream_peer, mut proxy_upstream) = tokio::io::duplex(64);

        let relay = tokio::spawn(async move {
            relay_tcp(&mut proxy_client, &mut proxy_upstream, &guard, &request).await
        });
        tokio::task::yield_now().await;

        engine
            .enter_fail_closed("candidate policy validation failed")
            .unwrap();

        tokio::time::timeout(std::time::Duration::from_secs(1), relay)
            .await
            .expect("raw relay should close when its generation becomes stale")
            .expect("relay task should not panic")
            .expect("stale relay closure should be clean");
    }

    /// Regression for WP-6 H5: the in-sandbox `session-create` POST crossed a
    /// `tls: skip` passthrough route while a provider-env-only settings poll
    /// activated credentials. The raw tunnel has no credential dependency, so
    /// it must carry the full exchange instead of closing as stale.
    #[tokio::test]
    async fn h5_passthrough_tunnel_survives_provider_env_only_activation() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let engine = Arc::new(OpaEngine::from_strings(POLICY_REGO, EMPTY_POLICY_DATA).unwrap());
        let decision = decision(engine.current_generation());
        let guard = prepare_raw_relay(None, &engine, &decision, CredentialDependency::Independent)
            .expect("current generation should pin a raw relay");
        assert_eq!(
            guard.credential_dependency(),
            CredentialDependency::Independent
        );
        let request = request_context();
        let (mut client_peer, mut proxy_client) = tokio::io::duplex(1024);
        let (mut upstream_peer, mut proxy_upstream) = tokio::io::duplex(1024);
        let relay_guard = guard.clone();
        let relay = tokio::spawn(async move {
            relay_tcp(
                &mut proxy_client,
                &mut proxy_upstream,
                &relay_guard,
                &request,
            )
            .await
        });
        tokio::task::yield_now().await;

        // Provider-env-only revision: same policy, new credential epoch.
        engine.activate_credentials(|| {}).unwrap();
        assert!(!guard.is_stale());

        let request_bytes = b"POST /api/v1/sessions HTTP/1.1\r\nHost: host.docker.internal\r\n\r\n";
        client_peer.write_all(request_bytes).await.unwrap();
        let mut received = vec![0; request_bytes.len()];
        upstream_peer.read_exact(&mut received).await.unwrap();
        assert_eq!(received, request_bytes);
        let response_bytes = b"HTTP/1.1 201 Created\r\ncontent-length: 0\r\n\r\n";
        upstream_peer.write_all(response_bytes).await.unwrap();
        let mut returned = vec![0; response_bytes.len()];
        client_peer.read_exact(&mut returned).await.unwrap();
        assert_eq!(returned, response_bytes);

        drop(client_peer);
        drop(upstream_peer);
        tokio::time::timeout(std::time::Duration::from_secs(5), relay)
            .await
            .expect("relay ends when both peers close")
            .expect("relay task should not panic")
            .expect("tunnel should complete cleanly");
        assert!(!guard.is_stale());
    }

    #[tokio::test]
    async fn credential_bound_raw_relay_closes_on_provider_env_activation() {
        let engine = Arc::new(OpaEngine::from_strings(POLICY_REGO, EMPTY_POLICY_DATA).unwrap());
        let decision = decision(engine.current_generation());
        let guard = prepare_raw_relay(None, &engine, &decision, CredentialDependency::Dependent)
            .expect("current generation should pin a raw relay");
        let request = request_context();
        // Peers stay open: only staleness can end this relay.
        let (_client_peer, mut proxy_client) = tokio::io::duplex(64);
        let (_upstream_peer, mut proxy_upstream) = tokio::io::duplex(64);
        let relay = tokio::spawn(async move {
            relay_tcp(&mut proxy_client, &mut proxy_upstream, &guard, &request).await
        });
        tokio::task::yield_now().await;

        engine.activate_credentials(|| {}).unwrap();

        tokio::time::timeout(std::time::Duration::from_secs(5), relay)
            .await
            .expect("credential-bound relay should close on a new credential epoch")
            .expect("relay task should not panic")
            .expect("stale relay closure should be clean");
    }

    #[test]
    fn credential_capable_http_relays_bind_the_credential_epoch() {
        let engine = OpaEngine::from_strings(POLICY_REGO, EMPTY_POLICY_DATA).unwrap();
        let decision = decision(engine.current_generation());
        let request = request_context();
        let context = prepare_http_relay(None, &engine, &decision, &request)
            .expect("current generation should prepare a relay");
        let PreparedHttpPolicy::Passthrough { generation_guard } = context.policy else {
            panic!("route-less relay should use a generation guard");
        };
        assert_eq!(
            generation_guard.credential_dependency(),
            CredentialDependency::Dependent,
            "passthrough HTTP rewrites credential placeholders"
        );
        let evaluator = pin_l7_evaluator(&engine, decision.policy_generation).unwrap();
        assert_eq!(evaluator.captured_credential_epoch(), Some(0));

        engine.activate_credentials(|| {}).unwrap();

        assert!(generation_guard.is_stale());
        assert!(evaluator.is_stale());
        // Relays prepared after the activation bind the new epoch.
        let context = prepare_http_relay(None, &engine, &decision, &request)
            .expect("policy is unchanged, so a new relay can start");
        let PreparedHttpPolicy::Passthrough { generation_guard } = context.policy else {
            panic!("route-less relay should use a generation guard");
        };
        assert_eq!(generation_guard.captured_credential_epoch(), Some(1));
        assert!(!generation_guard.is_stale());
    }

    #[tokio::test]
    async fn policy_only_raw_relay_still_closes_on_policy_reload() {
        let engine = Arc::new(OpaEngine::from_strings(POLICY_REGO, EMPTY_POLICY_DATA).unwrap());
        let decision = decision(engine.current_generation());
        let guard = prepare_raw_relay(None, &engine, &decision, CredentialDependency::Independent)
            .expect("current generation should pin a raw relay");
        let request = request_context();
        let (_client_peer, mut proxy_client) = tokio::io::duplex(64);
        let (_upstream_peer, mut proxy_upstream) = tokio::io::duplex(64);
        let relay = tokio::spawn(async move {
            relay_tcp(&mut proxy_client, &mut proxy_upstream, &guard, &request).await
        });
        tokio::task::yield_now().await;

        engine.reload(POLICY_REGO, EMPTY_POLICY_DATA).unwrap();

        tokio::time::timeout(std::time::Duration::from_secs(5), relay)
            .await
            .expect("an actual policy change still closes raw relays")
            .expect("relay task should not panic")
            .expect("stale relay closure should be clean");
    }
}
