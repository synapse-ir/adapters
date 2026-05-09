"""
SYNAPSE Adapter for John Snow Labs ner_clinical
================================================
Model:     ner_clinical (John Snow Labs Spark NLP for Healthcare)
Task:      Clinical Named Entity Recognition
Entities:  PROBLEM | TEST | TREATMENT
License:   Requires John Snow Labs Healthcare NLP license
Spec:      https://github.com/synapse-ir/spec
"""

from __future__ import annotations

from typing import Any

from synapse_sdk.base import AdapterBase
from synapse_sdk.types import CanonicalIR, ComplianceEnvelope, Entity


class NerClinicalAdapter(AdapterBase):
    """
    SYNAPSE adapter for the John Snow Labs ner_clinical model.

    ner_clinical extracts three entity types from clinical free text:
      - PROBLEM    — diagnoses, symptoms, and disease mentions
      - TEST       — lab tests, imaging studies, and diagnostic procedures
      - TREATMENT  — medications, procedures, and therapeutic interventions

    Trained on the i2b2 2010 clinical NLP challenge dataset.

    IMPORTANT: This adapter requires a valid John Snow Labs Healthcare NLP
    license. The model is not open source. See:
    https://www.johnsnowlabs.com/spark-nlp-health/

    Architecture note: This adapter follows the SYNAPSE pure-function contract.
    The Spark NLP pipeline must be built and called by the caller. The adapter
    prepares input for the pipeline (ingress) and processes its output (egress).
    The adapter itself makes no network calls and starts no Spark sessions.

    Typical caller usage:
        adapter = NerClinicalAdapter()
        model_input = adapter.ingress(ir)
        ner_output = spark_pipeline.transform(spark_df_from(model_input))
        result_ir  = adapter.egress(ner_chunks_from(ner_output), ir, latency_ms)
    """

    MODEL_ID = "johnsnowlabs/ner_clinical"
    ADAPTER_VERSION = "1.0.0"

    # Entity types produced by ner_clinical (exhaustive — no others exist)
    ENTITY_TYPES: frozenset[str] = frozenset({"PROBLEM", "TEST", "TREATMENT"})

    # PII-sensitive entity label — PROBLEM entities may contain patient info
    PII_ENTITY_TYPES: frozenset[str] = frozenset({"PROBLEM"})

    def ingress(self, ir: CanonicalIR) -> dict[str, Any]:
        """
        Convert canonical IR to ner_clinical native input format.

        ner_clinical expects plain text. The pipeline's DocumentAssembler
        takes a Spark DataFrame with a "text" column. This method extracts
        the text from the canonical IR payload.

        Args:
            ir: Canonical IR with payload.content containing clinical text.

        Returns:
            dict with key "text" containing the clinical text string.
            Returns {"text": ""} if payload.content is None or missing.
        """
        content = ""
        if ir.payload is not None and ir.payload.content is not None:
            content = ir.payload.content
        return {"text": content}

    def egress(
        self,
        model_output: Any,
        original_ir: CanonicalIR,
        latency_ms: int,
    ) -> CanonicalIR:
        """
        Convert ner_clinical output (ner_chunk annotations) to canonical IR.

        ner_clinical's NerConverter produces Annotation objects with:
          .result               — entity surface text
          .begin                — character start offset
          .end                  — character end offset
          .metadata["entity"]   — one of PROBLEM, TEST, TREATMENT
          .metadata["confidence"] — confidence score as a string

        model_output should be the collected ner_chunk annotations from the
        Spark pipeline result, passed as a Python list of Annotation objects.
        A non-list value (e.g. a dict from the validator harness) is treated
        as empty — the adapter must never raise on unexpected input shapes.

        When pii_present=True in the incoming compliance_envelope, this
        adapter propagates pii_present=True in the output IR (SYNAPSE G-S04).

        Args:
            model_output: List of Annotation objects from NerConverter ner_chunk.
                          Pass an empty list if no entities were found.
            original_ir:  The original canonical IR passed to ingress().
            latency_ms:   Wall-clock inference time in milliseconds.

        Returns:
            Updated canonical IR with entities populated and provenance appended.
        """
        annotations = model_output if isinstance(model_output, list) else []

        updated = original_ir.clone()
        entities: list[Entity] = []
        confidences: list[float] = []

        for annotation in annotations:
            entity_label = annotation.metadata.get("entity", "")

            # Skip any labels outside the three known entity types.
            # Defensive guard against unexpected model output.
            if entity_label not in self.ENTITY_TYPES:
                continue

            # confidence is stored as a string in metadata — must cast
            raw_conf = annotation.metadata.get("confidence", "0.0")
            try:
                conf = float(raw_conf)
            except (ValueError, TypeError):
                conf = 0.0

            # Clamp to valid range per SYNAPSE spec CONFIDENCE_RANGE rule
            conf = max(0.0, min(1.0, conf))
            confidences.append(conf)

            entities.append(
                Entity(
                    text=annotation.result,
                    label=entity_label,
                    start=annotation.begin,
                    end=annotation.end,
                    confidence=conf,
                )
            )

        # Populate entities in the IR payload
        updated.payload.entities = entities if entities else None

        # Compute overall confidence: mean of entity scores, or 0.0 if none
        overall_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        # G-S04: propagate pii_present when the incoming IR flags it, or when
        # PROBLEM entities are extracted (PROBLEM spans may contain patient info).
        # Only upgrade — never downgrade an existing True to None or False.
        incoming_pii = (
            original_ir.compliance_envelope is not None
            and original_ir.compliance_envelope.pii_present is True
        )
        has_problem_entities = any(
            e.label in self.PII_ENTITY_TYPES for e in entities
        )
        if (incoming_pii or has_problem_entities) and updated.compliance_envelope is not None:
            if not updated.compliance_envelope.pii_present:
                orig_env = original_ir.compliance_envelope
                updated.compliance_envelope = ComplianceEnvelope(
                    required_tags=orig_env.required_tags,
                    pii_present=True,
                    data_residency=orig_env.data_residency,
                    retention_policy=orig_env.retention_policy,
                    purpose_limitation=orig_env.purpose_limitation,
                )

        # Append provenance — MUST NOT modify existing entries
        updated.provenance.append(
            self.build_provenance(
                confidence=overall_confidence,
                latency_ms=latency_ms,
            )
        )

        return updated
