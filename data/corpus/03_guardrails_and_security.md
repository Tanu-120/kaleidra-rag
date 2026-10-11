# Guardrails and Security

Kaleidra applies guardrails at ingest, query time, retrieval time, and answer time.

## PII redaction before indexing

Personally identifiable information is detected with regex patterns and optional Presidio NER, then redacted before chunks are embedded. Supported patterns include email addresses, credit cards validated with the Luhn checksum, phone numbers (including +91), Indian PAN and Aadhaar-style IDs, IFSC codes, API key prefixes, and IP addresses. Because redaction runs at ingest, emails and card numbers never enter the vector store.

## Prompt injection defense

Direct injection in the user question is scored with weighted rule families (instruction override, DAN-style jailbreaks, system-prompt exfiltration). Risk at or above 0.6 blocks the pipeline before retrieval. Benign questions that merely mention the word "ignore" in a technical sense are allowed.

## Indirect injection in retrieved context

Documents can contain planted instructions aimed at the model. Kaleidra scans retrieved chunk text sentence-by-sentence. Suspicious sentences are replaced with a quarantine marker and the evidence card is labelled. This is the realistic RAG attack surface: poisoned knowledge, not only hostile user input.

## Output scanning

After generation, the full answer text is re-scanned for PII and for simple system-prompt leak heuristics. Any PII found in claims is redacted before the response is returned to the UI.

## Incident response for operators

If a poisoned document is discovered in the corpus, remove the file from `data/corpus`, call `POST /api/ingest` to rebuild indexes, and review History for answers that cited the quarantined source. Audit logs for the local service are the uvicorn access log and any eval reports written by `scripts/run_eval.py`.
