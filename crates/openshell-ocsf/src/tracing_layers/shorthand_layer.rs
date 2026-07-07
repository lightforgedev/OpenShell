// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tracing layer that writes OCSF shorthand to a writer.

use std::io::Write;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use chrono::Utc;
use tracing::Subscriber;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;

use crate::SeverityId;
use crate::tracing_layers::event_bridge::{OCSF_TARGET, clone_current_event};

/// A tracing `Layer` that intercepts OCSF events and writes shorthand output.
///
/// Events with `target: "ocsf"` are formatted via `format_shorthand()`.
/// Non-OCSF events are formatted with a simple fallback.
///
/// Each line is prefixed with a UTC timestamp (`YYYY-MM-DDTHH:MM:SS.mmmZ`)
/// since this layer writes directly to a file with no outer display layer
/// to supply timestamps.
pub struct OcsfShorthandLayer<W: Write + Send + 'static> {
    writer: Mutex<W>,
    /// Whether to include non-OCSF events in the output.
    include_non_ocsf: bool,
    min_ocsf_severity_rank: Arc<AtomicU8>,
}

impl<W: Write + Send + 'static> OcsfShorthandLayer<W> {
    /// Create a new shorthand layer writing to the given writer.
    #[must_use]
    pub fn new(writer: W) -> Self {
        Self {
            writer: Mutex::new(writer),
            include_non_ocsf: true,
            min_ocsf_severity_rank: Arc::new(AtomicU8::new(severity_rank(
                SeverityId::Informational,
            ))),
        }
    }

    /// Set whether non-OCSF tracing events should be included.
    #[must_use]
    pub fn with_non_ocsf(mut self, include: bool) -> Self {
        self.include_non_ocsf = include;
        self
    }

    /// Set the minimum OCSF event severity rendered into shorthand output.
    #[must_use]
    pub fn with_min_ocsf_severity(mut self, severity: SeverityId) -> Self {
        self.min_ocsf_severity_rank = Arc::new(AtomicU8::new(severity_rank(severity)));
        self
    }

    /// Set the shared minimum OCSF severity rank used by this layer.
    ///
    /// This lets the sandbox settings poll loop change shorthand verbosity at
    /// runtime without rebuilding the tracing subscriber.
    #[must_use]
    pub fn with_min_ocsf_severity_rank(mut self, rank: Arc<AtomicU8>) -> Self {
        self.min_ocsf_severity_rank = rank;
        self
    }
}

impl<S, W> Layer<S> for OcsfShorthandLayer<W>
where
    S: Subscriber,
    W: Write + Send + 'static,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let meta = event.metadata();

        let ts = Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ");

        if meta.target() == OCSF_TARGET {
            // This is an OCSF event — clone from thread-local (non-consuming)
            if let Some(ocsf_event) = clone_current_event() {
                if severity_rank(ocsf_event.base().severity)
                    < self.min_ocsf_severity_rank.load(Ordering::Relaxed)
                {
                    return;
                }
                let line = ocsf_event.format_shorthand();
                if let Ok(mut w) = self.writer.lock() {
                    let _ = writeln!(w, "{ts} OCSF {line}");
                }
            }
        } else if self.include_non_ocsf {
            // Fallback: format non-OCSF events with basic format
            let level = meta.level();
            let target = meta.target();
            // Extract message from the event fields
            let mut message = String::new();
            event.record(&mut MessageVisitor(&mut message));
            if let Ok(mut w) = self.writer.lock() {
                let _ = writeln!(w, "{ts} {level} {target}: {message}");
            }
        }
    }
}

/// Convert an OCSF severity into a monotonic comparison rank.
#[must_use]
pub fn severity_rank(severity: SeverityId) -> u8 {
    match severity {
        SeverityId::Unknown => 0,
        SeverityId::Informational => 1,
        SeverityId::Low => 2,
        SeverityId::Medium => 3,
        SeverityId::High => 4,
        SeverityId::Critical => 5,
        SeverityId::Fatal => 6,
        SeverityId::Other => 7,
    }
}

/// Simple visitor that extracts the message field from tracing events.
struct MessageVisitor<'a>(&'a mut String);

impl tracing::field::Visit for MessageVisitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            *self.0 = format!("{value:?}");
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            *self.0 = value.to_string();
        }
    }
}

/// Test helper: wraps `Arc<Mutex<Vec<u8>>>` so it implements `Write + Send`.
#[cfg(test)]
struct SyncWriter(Arc<Mutex<Vec<u8>>>);

#[cfg(test)]
impl Write for SyncWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.lock().unwrap().flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shorthand_layer_creation() {
        let buffer: Vec<u8> = Vec::new();
        let _layer = OcsfShorthandLayer::new(buffer);
    }

    #[test]
    fn test_shorthand_layer_with_non_ocsf_toggle() {
        let buffer: Vec<u8> = Vec::new();
        let layer = OcsfShorthandLayer::new(buffer).with_non_ocsf(false);
        assert!(!layer.include_non_ocsf);
    }

    #[test]
    fn test_shorthand_layer_with_min_ocsf_severity() {
        let buffer: Vec<u8> = Vec::new();
        let layer = OcsfShorthandLayer::new(buffer).with_min_ocsf_severity(SeverityId::Medium);
        assert_eq!(
            layer.min_ocsf_severity_rank.load(Ordering::Relaxed),
            severity_rank(SeverityId::Medium)
        );
    }

    #[test]
    fn test_non_ocsf_fallback_includes_timestamp() {
        use std::sync::Arc;
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;

        let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let writer = SyncWriter(buffer.clone());
        let layer = OcsfShorthandLayer::new(writer).with_non_ocsf(true);

        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = subscriber.set_default();

        tracing::info!("test message");

        let output = buffer.lock().unwrap();
        let line = String::from_utf8_lossy(&output);
        // Should start with a timestamp like 2026-04-01T...
        assert!(
            line.contains('T') && line.contains('Z'),
            "Expected timestamp in output, got: {line}"
        );
        assert!(
            line.contains("test message"),
            "Expected message, got: {line}"
        );
    }

    #[test]
    fn test_ocsf_severity_threshold_filters_info_and_keeps_medium() {
        use crate::events::base_event::BaseEventData;
        use crate::events::{BaseEvent, OcsfEvent};
        use crate::objects::{Metadata, Product};
        use std::sync::Arc;
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;

        fn event(severity: SeverityId) -> OcsfEvent {
            let mut base = BaseEventData::new(
                0,
                "Base Event",
                0,
                "Uncategorized",
                99,
                "Other",
                severity,
                Metadata {
                    version: "1.7.0".to_string(),
                    product: Product::openshell_sandbox("0.1.0"),
                    profiles: vec![],
                    uid: None,
                    log_source: None,
                },
            );
            base.set_message(format!("severity={}", severity.label()));
            OcsfEvent::Base(BaseEvent { base })
        }

        let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let writer = SyncWriter(buffer.clone());
        let layer = OcsfShorthandLayer::new(writer).with_min_ocsf_severity(SeverityId::Medium);

        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = subscriber.set_default();

        crate::ocsf_emit!(event(SeverityId::Informational));
        crate::ocsf_emit!(event(SeverityId::Medium));

        let output = buffer.lock().unwrap();
        let text = String::from_utf8_lossy(&output);
        assert!(!text.contains("severity=Informational"), "got: {text}");
        assert!(text.contains("severity=Medium"), "got: {text}");
    }

    #[test]
    fn test_ocsf_severity_threshold_can_change_at_runtime() {
        use crate::events::base_event::BaseEventData;
        use crate::events::{BaseEvent, OcsfEvent};
        use crate::objects::{Metadata, Product};
        use std::sync::Arc;
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;

        fn event(severity: SeverityId) -> OcsfEvent {
            let mut base = BaseEventData::new(
                0,
                "Base Event",
                0,
                "Uncategorized",
                99,
                "Other",
                severity,
                Metadata {
                    version: "1.7.0".to_string(),
                    product: Product::openshell_sandbox("0.1.0"),
                    profiles: vec![],
                    uid: None,
                    log_source: None,
                },
            );
            base.set_message(format!("severity={}", severity.label()));
            OcsfEvent::Base(BaseEvent { base })
        }

        let threshold = Arc::new(AtomicU8::new(severity_rank(SeverityId::Medium)));
        let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let writer = SyncWriter(buffer.clone());
        let layer = OcsfShorthandLayer::new(writer).with_min_ocsf_severity_rank(threshold.clone());

        let subscriber = tracing_subscriber::registry().with(layer);
        let _guard = subscriber.set_default();

        crate::ocsf_emit!(event(SeverityId::Informational));
        threshold.store(severity_rank(SeverityId::Informational), Ordering::Relaxed);
        crate::ocsf_emit!(event(SeverityId::Informational));

        let output = buffer.lock().unwrap();
        let text = String::from_utf8_lossy(&output);
        assert_eq!(
            text.matches("severity=Informational").count(),
            1,
            "got: {text}"
        );
    }
}
