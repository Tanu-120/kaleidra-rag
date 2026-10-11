# Nimbus Platform Overview

Nimbus is a managed event-streaming platform built by Nimbus Labs. It lets teams publish and consume events with low latency. Nimbus runs on Kubernetes and stores data in the Nimbus Log, a distributed append-only storage layer.

## Architecture

The Nimbus Gateway receives all client traffic and forwards it to a Broker Cluster. Each Broker Cluster is coordinated by the Nimbus Controller, which tracks partition leadership using the Raft consensus protocol. Messages are replicated to three brokers before an acknowledgement is returned to the producer.

The Schema Registry validates every event against a registered Avro or JSON Schema. Events that fail validation are routed to the Dead Letter Queue for later inspection.

## Retries and Delivery Guarantees

Producers retry failed sends up to five times using exponential backoff starting at 100 milliseconds. The retry mechanism doubles the delay after every failed attempt and adds random jitter of up to twenty percent to avoid thundering herds. Nimbus offers at-least-once delivery by default and exactly-once delivery when idempotent producers are enabled.

Consumers commit offsets to the Nimbus Log. If a consumer crashes, the Controller reassigns its partitions to a healthy consumer in the same Consumer Group within thirty seconds.

## Observability

Nimbus exports metrics in Prometheus format on port 9102. The key metrics are consumer lag, broker disk usage and request latency. Alerts should fire when consumer lag exceeds ten thousand messages for more than five minutes. Traces use OpenTelemetry and are sent to the collector configured in the Nimbus Gateway.
