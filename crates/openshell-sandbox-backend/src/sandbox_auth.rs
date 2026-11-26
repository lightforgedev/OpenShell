// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral authentication for `OpenShell` Sandbox Protocol connections.

use std::sync::Mutex;

use openshell_core::jwt::{
    AuthenticatedSandboxSession, CredentialEpoch, SandboxId, SessionJwtError, SessionJwtVerifier,
};
use openshell_core::sandbox_generation::SandboxGenerationId;
use tonic::metadata::MetadataMap;
use uuid::Uuid;

/// Server-local identity assigned after a byte stream completes TLS.
///
/// It cannot be supplied by a compute driver or workload. Protocol handlers use
/// it to bind authenticated requests to the connection that performed attach.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SandboxConnectionId(Uuid);

impl SandboxConnectionId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for SandboxConnectionId {
    fn default() -> Self {
        Self::new()
    }
}

/// Principal returned only after strict bearer validation and identity binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxProtocolPrincipal {
    connection_id: SandboxConnectionId,
    session: AuthenticatedSandboxSession,
}

impl SandboxProtocolPrincipal {
    #[must_use]
    pub const fn connection_id(&self) -> SandboxConnectionId {
        self.connection_id
    }

    #[must_use]
    pub const fn session(&self) -> &AuthenticatedSandboxSession {
        &self.session
    }
}

/// Validates Sandbox Protocol metadata without depending on its byte transport.
pub struct SandboxProtocolAuthenticator {
    verifier: SessionJwtVerifier,
    expected_sandbox_id: SandboxId,
    expected_runtime_generation: SandboxGenerationId,
    expected_auth_epoch: CredentialEpoch,
}

impl std::fmt::Debug for SandboxProtocolAuthenticator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SandboxProtocolAuthenticator")
            .field("verifier", &self.verifier)
            .field("expected_sandbox_id", &self.expected_sandbox_id)
            .field(
                "expected_runtime_generation",
                &self.expected_runtime_generation,
            )
            .field("expected_auth_epoch", &self.expected_auth_epoch)
            .finish()
    }
}

impl SandboxProtocolAuthenticator {
    #[must_use]
    pub const fn new(
        verifier: SessionJwtVerifier,
        expected_sandbox_id: SandboxId,
        expected_runtime_generation: SandboxGenerationId,
        expected_auth_epoch: CredentialEpoch,
    ) -> Self {
        Self {
            verifier,
            expected_sandbox_id,
            expected_runtime_generation,
            expected_auth_epoch,
        }
    }

    pub fn authenticate(
        &self,
        connection_id: SandboxConnectionId,
        metadata: &MetadataMap,
    ) -> Result<SandboxProtocolPrincipal, SandboxAuthError> {
        let mut values = metadata.get_all("authorization").iter();
        let value = values.next().ok_or(SandboxAuthError::MissingBearer)?;
        if values.next().is_some() {
            return Err(SandboxAuthError::DuplicateBearer);
        }
        let value = value
            .to_str()
            .map_err(|_| SandboxAuthError::InvalidBearer)?;
        let token = value
            .strip_prefix("Bearer ")
            .filter(|token| !token.is_empty() && !token.chars().any(char::is_whitespace))
            .ok_or(SandboxAuthError::InvalidBearer)?;
        let session = self.verifier.verify(token)?;
        if session.sandbox_id != self.expected_sandbox_id {
            return Err(SandboxAuthError::WrongSandbox);
        }
        if session.runtime_generation != self.expected_runtime_generation {
            return Err(SandboxAuthError::WrongRuntimeGeneration);
        }
        if session.auth_epoch != self.expected_auth_epoch {
            return Err(SandboxAuthError::StaleCredentialEpoch);
        }
        Ok(SandboxProtocolPrincipal {
            connection_id,
            session,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ActiveConnection {
    id: SandboxConnectionId,
    epoch: CredentialEpoch,
}

#[derive(Debug)]
struct ConnectionState {
    active: Option<ActiveConnection>,
    pending: Option<ActiveConnection>,
    highest_epoch: Option<CredentialEpoch>,
    supervisor_instance_id: Option<crate::boundary_protocol::SupervisorInstanceId>,
    /// Current driver-issued supervisor generation. `None` until a
    /// generation-aware supervisor attaches.
    supervisor_generation: Option<u64>,
    /// Connections of a fenced (superseded) supervisor generation, kept so
    /// their requests are rejected with an explicit cause. Bounded; an
    /// evicted entry is still rejected as not attached.
    fenced: std::collections::VecDeque<(SandboxConnectionId, u64)>,
    terminal: bool,
}

/// Fenced connections retained for explicit rejection causes.
const MAX_FENCED_CONNECTIONS: usize = 64;

impl ConnectionState {
    fn fenced_generation(&self, connection_id: SandboxConnectionId) -> Option<u64> {
        self.fenced
            .iter()
            .find(|(id, _)| *id == connection_id)
            .map(|(_, generation)| *generation)
    }

    fn fence(&mut self, connection_id: SandboxConnectionId, generation: u64) {
        if self.fenced.len() >= MAX_FENCED_CONNECTIONS {
            self.fenced.pop_front();
        }
        self.fenced.push_back((connection_id, generation));
    }

    fn reject_fenced(&self, connection_id: SandboxConnectionId) -> Result<(), SandboxAuthError> {
        self.fenced_generation(connection_id)
            .map_or(Ok(()), |fenced| {
                Err(SandboxAuthError::FencedSupervisorGeneration {
                    fenced,
                    current: self.supervisor_generation.unwrap_or(0),
                })
            })
    }
}

/// Result of staging an attach.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AttachOutcome {
    /// An older unconfirmed candidate that may be closed.
    pub replaced_pending: Option<SandboxConnectionId>,
    /// The confirmed connection of a superseded supervisor generation. The
    /// caller must treat it as lost supervision and close it.
    pub fenced_active: Option<SandboxConnectionId>,
    /// Supervisor generation current before this attach (0 = legacy).
    pub previous_generation: Option<u64>,
    /// Supervisor generation current after this attach.
    pub generation: Option<u64>,
}

/// Enforces the one-active-connection and monotonically increasing epoch rules.
#[derive(Debug)]
pub struct SandboxConnectionRegistry {
    state: Mutex<ConnectionState>,
}

impl SandboxConnectionRegistry {
    #[must_use]
    pub fn new(
        _session_id: openshell_core::SandboxSessionId,
        _session_rotation: openshell_core::jwt::SessionRotation,
    ) -> Self {
        Self {
            state: Mutex::new(ConnectionState {
                active: None,
                pending: None,
                highest_epoch: None,
                supervisor_instance_id: None,
                supervisor_generation: None,
                fenced: std::collections::VecDeque::new(),
                terminal: false,
            }),
        }
    }

    /// Stage a fully authenticated connection for confirmation. The returned
    /// ID identifies an older unconfirmed candidate that may be closed. The
    /// current active connection remains authoritative until [`Self::confirm`]
    /// promotes this candidate.
    ///
    /// This is the legacy entry point for supervisors that present no
    /// driver-issued generation.
    pub fn attach(
        &self,
        principal: &SandboxProtocolPrincipal,
        supervisor_instance_id: crate::boundary_protocol::SupervisorInstanceId,
    ) -> Result<Option<SandboxConnectionId>, SandboxAuthError> {
        self.attach_supervisor(principal, supervisor_instance_id, None)
            .map(|outcome| outcome.replaced_pending)
    }

    /// Stage an authenticated connection, applying the supervisor generation
    /// handover rules:
    ///
    /// - a generation lower than the current one is stale and rejected;
    /// - the current generation may reconnect only as the same process;
    /// - a higher generation fences the current one first: its active and
    ///   pending connections are revoked (and reported for closing), and any
    ///   later request from them is rejected, before the new connection is
    ///   staged.
    ///
    /// Every check runs before any state changes, so a rejected attach never
    /// fences the current supervisor.
    pub fn attach_supervisor(
        &self,
        principal: &SandboxProtocolPrincipal,
        supervisor_instance_id: crate::boundary_protocol::SupervisorInstanceId,
        supervisor_generation: Option<&crate::boundary_protocol::SupervisorGeneration>,
    ) -> Result<AttachOutcome, SandboxAuthError> {
        let epoch = principal.session.auth_epoch;
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.terminal {
            return Err(SandboxAuthError::TerminalSession);
        }
        state.reject_fenced(principal.connection_id)?;
        let requested = match supervisor_generation {
            Some(generation) => {
                if generation.generation == 0
                    || generation.sandbox_id != principal.session.sandbox_id.as_str()
                    || generation.auth_epoch != epoch
                {
                    return Err(SandboxAuthError::SupervisorGenerationMismatch);
                }
                Some(generation.generation)
            }
            None => None,
        };
        let previous_generation = state.supervisor_generation;
        let handover = match (previous_generation, requested) {
            (Some(current), None) => {
                return Err(SandboxAuthError::MissingSupervisorGeneration { current });
            }
            (Some(current), Some(requested)) if requested < current => {
                return Err(SandboxAuthError::StaleSupervisorGeneration { current, requested });
            }
            (Some(current), Some(requested)) if requested == current => {
                if state.supervisor_instance_id != Some(supervisor_instance_id) {
                    return Err(SandboxAuthError::WrongSupervisorInstance);
                }
                false
            }
            // A higher generation, or the first generation-aware supervisor
            // after a legacy one, replaces whoever is bound.
            (_, Some(_)) => state.supervisor_instance_id.is_some(),
            (None, None) => match state.supervisor_instance_id {
                Some(expected) if expected != supervisor_instance_id => {
                    return Err(SandboxAuthError::WrongSupervisorInstance);
                }
                _ => false,
            },
        };
        if !handover {
            if let Some(active) = state.active {
                if active.id == principal.connection_id && active.epoch == epoch {
                    return Ok(AttachOutcome {
                        previous_generation,
                        generation: previous_generation,
                        ..AttachOutcome::default()
                    });
                }
                if epoch <= active.epoch {
                    return Err(SandboxAuthError::StaleCredentialEpoch);
                }
            }
            if let Some(pending) = state.pending {
                if pending.id == principal.connection_id && pending.epoch == epoch {
                    return Ok(AttachOutcome {
                        previous_generation,
                        generation: previous_generation,
                        ..AttachOutcome::default()
                    });
                }
                if epoch < pending.epoch {
                    return Err(SandboxAuthError::StaleCredentialEpoch);
                }
            }
        }
        if state.highest_epoch.is_some_and(|highest| epoch < highest) {
            return Err(SandboxAuthError::StaleCredentialEpoch);
        }

        // All checks passed; commit.
        let mut outcome = AttachOutcome {
            previous_generation,
            generation: requested.or(previous_generation),
            ..AttachOutcome::default()
        };
        if handover {
            let fenced_generation = previous_generation.unwrap_or(0);
            if let Some(active) = state.active.take() {
                state.fence(active.id, fenced_generation);
                outcome.fenced_active = Some(active.id);
            }
            if let Some(pending) = state.pending.take() {
                state.fence(pending.id, fenced_generation);
                outcome.replaced_pending = Some(pending.id);
            }
        } else {
            outcome.replaced_pending = state.pending.map(|pending| pending.id);
        }
        state.supervisor_instance_id = Some(supervisor_instance_id);
        if requested.is_some() {
            state.supervisor_generation = requested;
        }
        state.highest_epoch = Some(epoch);
        state.pending = Some(ActiveConnection {
            id: principal.connection_id,
            epoch,
        });
        Ok(outcome)
    }

    /// Current driver-issued supervisor generation (`None` for legacy).
    #[must_use]
    pub fn supervisor_generation(&self) -> Option<u64> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .supervisor_generation
    }

    /// Promote an attached, confirmed candidate to the active connection. The
    /// returned ID is the previously active connection, which may now be
    /// closed without creating an unsupervised interval.
    pub fn confirm(
        &self,
        principal: &SandboxProtocolPrincipal,
    ) -> Result<Option<SandboxConnectionId>, SandboxAuthError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.terminal {
            return Err(SandboxAuthError::TerminalSession);
        }
        state.reject_fenced(principal.connection_id)?;
        if state
            .active
            .is_some_and(|active| active.id == principal.connection_id)
        {
            return Ok(None);
        }
        let pending = state
            .pending
            .filter(|pending| pending.id == principal.connection_id)
            .ok_or(SandboxAuthError::ConnectionNotAttached)?;
        let replaced = state.active.map(|active| active.id);
        state.active = Some(pending);
        state.pending = None;
        Ok(replaced)
    }

    /// Confirm may run on either the active connection (an idempotent replay)
    /// or its staged replacement. Other operations require the active one.
    pub fn require_attached(
        &self,
        principal: &SandboxProtocolPrincipal,
    ) -> Result<(), SandboxAuthError> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.terminal {
            return Err(SandboxAuthError::TerminalSession);
        }
        state.reject_fenced(principal.connection_id)?;
        if state
            .active
            .is_some_and(|active| active.id == principal.connection_id)
            || state
                .pending
                .is_some_and(|pending| pending.id == principal.connection_id)
        {
            Ok(())
        } else {
            Err(SandboxAuthError::ConnectionNotAttached)
        }
    }

    pub fn require_active(
        &self,
        principal: &SandboxProtocolPrincipal,
    ) -> Result<(), SandboxAuthError> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.terminal {
            return Err(SandboxAuthError::TerminalSession);
        }
        state.reject_fenced(principal.connection_id)?;
        if state
            .active
            .is_none_or(|active| active.id != principal.connection_id)
        {
            return Err(SandboxAuthError::ConnectionNotAttached);
        }
        Ok(())
    }

    /// Remove a physical connection. Returns `true` only when it was the
    /// confirmed active connection and recovery must begin.
    pub fn disconnect(&self, connection_id: SandboxConnectionId) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.fenced.retain(|(id, _)| *id != connection_id);
        let was_active = state
            .active
            .is_some_and(|active| active.id == connection_id);
        if was_active {
            state.active = None;
        }
        if state
            .pending
            .is_some_and(|pending| pending.id == connection_id)
        {
            state.pending = None;
        }
        was_active
    }

    pub fn mark_terminal(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.terminal = true;
        state.active = None;
        state.pending = None;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SandboxAuthError {
    #[error("authorization metadata is missing")]
    MissingBearer,
    #[error("authorization metadata must occur exactly once")]
    DuplicateBearer,
    #[error("authorization metadata is not a valid bearer credential")]
    InvalidBearer,
    #[error("sandbox JWT validation failed: {0}")]
    Jwt(#[from] SessionJwtError),
    #[error("authenticated sandbox identity does not match this runtime")]
    WrongSandbox,
    #[error("authenticated runtime generation does not match this sandbox runtime")]
    WrongRuntimeGeneration,
    #[error("Sandbox Protocol credential epoch is stale or already active")]
    StaleCredentialEpoch,
    #[error("sandbox runtime is already bound to another supervisor process")]
    WrongSupervisorInstance,
    #[error("supervisor generation {requested} is stale; generation {current} is current")]
    StaleSupervisorGeneration { current: u64, requested: u64 },
    #[error("supervisor generation {current} is current; attach must present a generation")]
    MissingSupervisorGeneration { current: u64 },
    #[error("supervisor generation is not bound to this sandbox and credential epoch")]
    SupervisorGenerationMismatch,
    #[error("supervisor generation {fenced} was fenced by generation {current}")]
    FencedSupervisorGeneration { fenced: u64, current: u64 },
    #[error("Sandbox Protocol connection has not completed attach")]
    ConnectionNotAttached,
    #[error("sandbox session is terminal")]
    TerminalSession,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openshell_core::jwt::{
        DEFAULT_SESSION_TOKEN_TTL, JwtClock, SandboxRuntimeIdentity, SessionJwtIssuer,
        SessionTokenProfile, SessionVerificationKey,
    };
    use openshell_core::sandbox_generation::SandboxGenerationId;
    use rcgen::{KeyPair, PKCS_ED25519};

    use super::*;

    #[derive(Debug)]
    struct FixedClock;

    impl JwtClock for FixedClock {
        fn now_unix_seconds(&self) -> i64 {
            1_900_000_000
        }
    }

    fn fixture(
        epoch: u64,
    ) -> (
        SandboxProtocolAuthenticator,
        openshell_core::jwt::MintedSessionToken,
    ) {
        let key = KeyPair::generate_for(&PKCS_ED25519).expect("generate key");
        let public_key_pem = key.public_key_pem().into_bytes();
        let clock: Arc<dyn JwtClock> = Arc::new(FixedClock);
        let sandbox_id = SandboxId::parse("sandbox-a").expect("sandbox ID");
        let issuer = SessionJwtIssuer::from_ed25519_pem(
            key.serialize_pem().as_bytes(),
            "current",
            "test",
            DEFAULT_SESSION_TOKEN_TTL,
            clock.clone(),
        )
        .expect("issuer");
        let verifier = SessionJwtVerifier::new(
            "test",
            SessionTokenProfile::Sandbox,
            [SessionVerificationKey {
                key_id: "current".to_string(),
                public_key_pem,
            }],
            clock,
        )
        .expect("verifier");
        let token = issuer
            .mint_pair(&SandboxRuntimeIdentity {
                sandbox_id: sandbox_id.clone(),
                runtime_generation: SandboxGenerationId::parse("generation-1")
                    .expect("runtime generation"),
                auth_epoch: CredentialEpoch::new(epoch).expect("epoch"),
            })
            .expect("token pair")
            .sandbox;
        (
            SandboxProtocolAuthenticator::new(
                verifier,
                sandbox_id,
                SandboxGenerationId::parse("generation-1").expect("runtime generation"),
                CredentialEpoch::new(epoch).expect("epoch"),
            ),
            token,
        )
    }

    fn registry_for(_principal: &SandboxProtocolPrincipal) -> SandboxConnectionRegistry {
        SandboxConnectionRegistry::new(
            openshell_core::SandboxSessionId::new(),
            openshell_core::jwt::SessionRotation::new(1).expect("rotation"),
        )
    }

    fn metadata(token: &str) -> MetadataMap {
        let mut metadata = MetadataMap::new();
        metadata.insert(
            "authorization",
            format!("Bearer {token}").parse().expect("metadata value"),
        );
        metadata
    }

    #[test]
    fn bearer_metadata_must_occur_exactly_once() {
        let (authenticator, token) = fixture(1);
        assert_eq!(
            authenticator.authenticate(SandboxConnectionId::new(), &MetadataMap::new()),
            Err(SandboxAuthError::MissingBearer)
        );
        let mut duplicate = metadata(token.token.expose_secret());
        duplicate.append(
            "authorization",
            format!("Bearer {}", token.token.expose_secret())
                .parse()
                .expect("metadata value"),
        );
        assert_eq!(
            authenticator.authenticate(SandboxConnectionId::new(), &duplicate),
            Err(SandboxAuthError::DuplicateBearer)
        );
    }

    #[test]
    fn reconnect_requires_disconnect_and_terminal_is_final() {
        let (first_authenticator, first_token) = fixture(1);
        let first_id = SandboxConnectionId::new();
        let first = first_authenticator
            .authenticate(first_id, &metadata(first_token.token.expose_secret()))
            .expect("first principal");
        let registry = registry_for(&first);
        let instance = crate::boundary_protocol::SupervisorInstanceId::new();
        assert_eq!(registry.attach(&first, instance), Ok(None));
        assert_eq!(registry.attach(&first, instance), Ok(None));
        assert_eq!(registry.confirm(&first), Ok(None));
        assert_eq!(registry.confirm(&first), Ok(None));

        assert!(registry.disconnect(first_id));
        assert_eq!(registry.attach(&first, instance), Ok(None));
        assert_eq!(registry.confirm(&first), Ok(None));
        registry.mark_terminal();
        assert_eq!(
            registry.require_active(&first),
            Err(SandboxAuthError::TerminalSession)
        );
        assert_eq!(
            registry.attach(&first, instance),
            Err(SandboxAuthError::TerminalSession)
        );
    }

    fn supervisor_generation(
        epoch: u64,
        generation: u64,
    ) -> crate::boundary_protocol::SupervisorGeneration {
        crate::boundary_protocol::SupervisorGeneration {
            sandbox_id: "sandbox-a".to_string(),
            auth_epoch: CredentialEpoch::new(epoch).expect("epoch"),
            generation,
        }
    }

    fn principal(
        authenticator: &SandboxProtocolAuthenticator,
        token: &openshell_core::jwt::MintedSessionToken,
    ) -> SandboxProtocolPrincipal {
        authenticator
            .authenticate(
                SandboxConnectionId::new(),
                &metadata(token.token.expose_secret()),
            )
            .expect("principal")
    }

    #[test]
    fn next_generation_fences_confirmed_generation_before_staging() {
        let (authenticator, token) = fixture(1);
        let first = principal(&authenticator, &token);
        let registry = registry_for(&first);
        let first_instance = crate::boundary_protocol::SupervisorInstanceId::new();
        let outcome = registry
            .attach_supervisor(&first, first_instance, Some(&supervisor_generation(1, 1)))
            .expect("attach generation 1");
        assert_eq!(outcome.fenced_active, None);
        assert_eq!(outcome.generation, Some(1));
        registry.confirm(&first).expect("confirm generation 1");

        // N+1 attaches while N is still the confirmed connection: N is
        // fenced first, then N+1 is staged.
        let replacement = principal(&authenticator, &token);
        let outcome = registry
            .attach_supervisor(
                &replacement,
                crate::boundary_protocol::SupervisorInstanceId::new(),
                Some(&supervisor_generation(1, 2)),
            )
            .expect("attach generation 2");
        assert_eq!(outcome.fenced_active, Some(first.connection_id));
        assert_eq!(outcome.previous_generation, Some(1));
        assert_eq!(outcome.generation, Some(2));
        assert_eq!(
            registry.require_active(&first),
            Err(SandboxAuthError::FencedSupervisorGeneration {
                fenced: 1,
                current: 2
            })
        );
        assert_eq!(
            registry.require_active(&replacement),
            Err(SandboxAuthError::ConnectionNotAttached),
            "generation 2 is not active until it confirms"
        );
        // Transport loss of the fenced connection is not a new recovery.
        assert!(!registry.disconnect(first.connection_id));
        assert_eq!(registry.confirm(&replacement), Ok(None));
        registry
            .require_active(&replacement)
            .expect("generation 2 active");
    }

    #[test]
    fn fenced_generation_cannot_reattach_or_confirm() {
        let (authenticator, token) = fixture(1);
        let first = principal(&authenticator, &token);
        let registry = registry_for(&first);
        let first_instance = crate::boundary_protocol::SupervisorInstanceId::new();
        registry
            .attach_supervisor(&first, first_instance, Some(&supervisor_generation(1, 1)))
            .expect("attach generation 1");
        registry.confirm(&first).expect("confirm generation 1");
        let replacement = principal(&authenticator, &token);
        registry
            .attach_supervisor(
                &replacement,
                crate::boundary_protocol::SupervisorInstanceId::new(),
                Some(&supervisor_generation(1, 2)),
            )
            .expect("attach generation 2");

        // A stale-generation Confirm on the fenced connection.
        assert_eq!(
            registry.confirm(&first),
            Err(SandboxAuthError::FencedSupervisorGeneration {
                fenced: 1,
                current: 2
            })
        );
        // N reconnects on a new transport after N+1 was accepted.
        let reconnect = principal(&authenticator, &token);
        assert_eq!(
            registry.attach_supervisor(
                &reconnect,
                first_instance,
                Some(&supervisor_generation(1, 1))
            ),
            Err(SandboxAuthError::StaleSupervisorGeneration {
                current: 2,
                requested: 1
            })
        );
        // A legacy (generation-less) attach cannot displace it either.
        assert_eq!(
            registry.attach(&reconnect, first_instance),
            Err(SandboxAuthError::MissingSupervisorGeneration { current: 2 })
        );
        registry
            .confirm(&replacement)
            .expect("generation 2 confirms");
    }

    #[test]
    fn next_generation_fences_unconfirmed_generation_and_rejects_its_late_confirm() {
        let (authenticator, token) = fixture(1);
        let first = principal(&authenticator, &token);
        let registry = registry_for(&first);
        let first_instance = crate::boundary_protocol::SupervisorInstanceId::new();
        registry
            .attach_supervisor(&first, first_instance, Some(&supervisor_generation(1, 1)))
            .expect("attach generation 1");

        // N+1 attaches between N's Attach and N's Confirm: N's pending
        // candidate is fenced and reported for closing.
        let replacement = principal(&authenticator, &token);
        let outcome = registry
            .attach_supervisor(
                &replacement,
                crate::boundary_protocol::SupervisorInstanceId::new(),
                Some(&supervisor_generation(1, 2)),
            )
            .expect("attach generation 2");
        assert_eq!(outcome.fenced_active, None);
        assert_eq!(outcome.replaced_pending, Some(first.connection_id));

        // N's Confirm arrives after N+1 was accepted.
        assert_eq!(
            registry.confirm(&first),
            Err(SandboxAuthError::FencedSupervisorGeneration {
                fenced: 1,
                current: 2
            })
        );
        assert_eq!(
            registry.require_attached(&first),
            Err(SandboxAuthError::FencedSupervisorGeneration {
                fenced: 1,
                current: 2
            })
        );
        // N's own retry on its fenced connection cannot re-stage it.
        assert_eq!(
            registry.attach_supervisor(&first, first_instance, Some(&supervisor_generation(1, 1))),
            Err(SandboxAuthError::FencedSupervisorGeneration {
                fenced: 1,
                current: 2
            })
        );
        assert_eq!(registry.confirm(&replacement), Ok(None));
        registry
            .require_active(&replacement)
            .expect("generation 2 active");
    }

    #[test]
    fn current_generation_reconnects_only_as_same_process() {
        let (authenticator, token) = fixture(1);
        let first = principal(&authenticator, &token);
        let registry = registry_for(&first);
        let instance = crate::boundary_protocol::SupervisorInstanceId::new();
        registry
            .attach_supervisor(&first, instance, Some(&supervisor_generation(1, 1)))
            .expect("attach");
        registry.confirm(&first).expect("confirm");
        assert!(registry.disconnect(first.connection_id));

        let impostor = principal(&authenticator, &token);
        assert_eq!(
            registry.attach_supervisor(
                &impostor,
                crate::boundary_protocol::SupervisorInstanceId::new(),
                Some(&supervisor_generation(1, 1)),
            ),
            Err(SandboxAuthError::WrongSupervisorInstance),
            "a second process cannot claim the current generation"
        );
        let reconnect = principal(&authenticator, &token);
        let outcome = registry
            .attach_supervisor(&reconnect, instance, Some(&supervisor_generation(1, 1)))
            .expect("same process reconnects");
        assert_eq!(outcome.fenced_active, None);
        registry.confirm(&reconnect).expect("reconfirm");
    }

    #[test]
    fn generation_must_be_bound_to_sandbox_and_epoch() {
        let (authenticator, token) = fixture(2);
        let first = principal(&authenticator, &token);
        let registry = registry_for(&first);
        let instance = crate::boundary_protocol::SupervisorInstanceId::new();
        let mut other_sandbox = supervisor_generation(2, 1);
        other_sandbox.sandbox_id = "sandbox-b".to_string();
        for generation in [
            other_sandbox,
            supervisor_generation(1, 1),
            supervisor_generation(2, 0),
        ] {
            assert_eq!(
                registry.attach_supervisor(&first, instance, Some(&generation)),
                Err(SandboxAuthError::SupervisorGenerationMismatch)
            );
        }
        assert_eq!(
            registry.supervisor_generation(),
            None,
            "rejections change nothing"
        );
    }

    #[test]
    fn rejected_handover_does_not_fence_current_generation() {
        let (first_authenticator, first_token) = fixture(2);
        let first = principal(&first_authenticator, &first_token);
        let registry = registry_for(&first);
        registry
            .attach_supervisor(
                &first,
                crate::boundary_protocol::SupervisorInstanceId::new(),
                Some(&supervisor_generation(2, 1)),
            )
            .expect("attach");
        registry.confirm(&first).expect("confirm");

        // A higher generation under an older credential epoch is stale.
        let (old_authenticator, old_token) = fixture(1);
        let stale = principal(&old_authenticator, &old_token);
        assert_eq!(
            registry.attach_supervisor(
                &stale,
                crate::boundary_protocol::SupervisorInstanceId::new(),
                Some(&supervisor_generation(1, 2)),
            ),
            Err(SandboxAuthError::StaleCredentialEpoch)
        );
        registry
            .require_active(&first)
            .expect("current generation stays active");
        assert_eq!(registry.supervisor_generation(), Some(1));
    }

    #[test]
    fn generation_handover_replaces_legacy_supervisor() {
        let (authenticator, token) = fixture(1);
        let first = principal(&authenticator, &token);
        let registry = registry_for(&first);
        registry
            .attach(
                &first,
                crate::boundary_protocol::SupervisorInstanceId::new(),
            )
            .expect("legacy attach");
        registry.confirm(&first).expect("legacy confirm");
        let replacement = principal(&authenticator, &token);
        let outcome = registry
            .attach_supervisor(
                &replacement,
                crate::boundary_protocol::SupervisorInstanceId::new(),
                Some(&supervisor_generation(1, 1)),
            )
            .expect("generation-aware handover");
        assert_eq!(outcome.fenced_active, Some(first.connection_id));
        assert_eq!(
            registry.require_active(&first),
            Err(SandboxAuthError::FencedSupervisorGeneration {
                fenced: 0,
                current: 1
            })
        );
    }

    #[test]
    fn replacement_supervisor_cannot_resume_existing_runtime_generation() {
        let (authenticator, token) = fixture(1);
        let first_id = SandboxConnectionId::new();
        let first = authenticator
            .authenticate(first_id, &metadata(token.token.expose_secret()))
            .expect("first principal");
        let registry = registry_for(&first);
        let first_instance = crate::boundary_protocol::SupervisorInstanceId::new();
        registry
            .attach(&first, first_instance)
            .expect("attach first supervisor");
        registry.confirm(&first).expect("confirm first supervisor");
        assert!(registry.disconnect(first_id));

        let replacement_id = SandboxConnectionId::new();
        let replacement = authenticator
            .authenticate(replacement_id, &metadata(token.token.expose_secret()))
            .expect("replacement principal");
        assert_eq!(
            registry.attach(
                &replacement,
                crate::boundary_protocol::SupervisorInstanceId::new(),
            ),
            Err(SandboxAuthError::WrongSupervisorInstance)
        );
        assert_eq!(registry.attach(&replacement, first_instance), Ok(None));
    }

    #[test]
    fn replacement_does_not_displace_active_connection_before_confirm() {
        let (first_authenticator, first_token) = fixture(1);
        let first_id = SandboxConnectionId::new();
        let first = first_authenticator
            .authenticate(first_id, &metadata(first_token.token.expose_secret()))
            .expect("first principal");
        let (replacement_authenticator, replacement_token) = fixture(2);
        let replacement_id = SandboxConnectionId::new();
        let replacement = replacement_authenticator
            .authenticate(
                replacement_id,
                &metadata(replacement_token.token.expose_secret()),
            )
            .expect("replacement principal");
        let registry = registry_for(&first);
        let instance = crate::boundary_protocol::SupervisorInstanceId::new();
        registry.attach(&first, instance).expect("attach first");
        registry.confirm(&first).expect("confirm first");

        assert_eq!(registry.attach(&replacement, instance), Ok(None));
        registry
            .require_active(&first)
            .expect("first remains active");
        assert_eq!(
            registry.require_active(&replacement),
            Err(SandboxAuthError::ConnectionNotAttached)
        );
        registry
            .require_attached(&replacement)
            .expect("replacement may confirm");
        assert_eq!(registry.confirm(&replacement), Ok(Some(first_id)));
        assert_eq!(
            registry.require_active(&first),
            Err(SandboxAuthError::ConnectionNotAttached)
        );
        registry
            .require_active(&replacement)
            .expect("replacement became active");
    }
}
