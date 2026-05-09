"""
SYNAPSE Adapter for docling-project/docling
===========================================
Model:     docling-project/docling
Task:      Document structure extraction and markdown conversion
Domains:   document, general
License:   MIT
Spec:      https://github.com/synapse-ir/spec
"""

from __future__ import annotations

from typing import Any

from synapse_sdk.base import AdapterBase
from synapse_sdk.types import CanonicalIR, Entity


class DoclingAdapter(AdapterBase):
    """
    SYNAPSE adapter for the Docling document AI model.

    Docling converts PDF, DOCX, PPTX, HTML, and other document formats into
    structured DoclingDocument objects with typed text blocks, tables, pictures,
    and rich layout metadata. This adapter bridges DoclingDocument output into
    the SYNAPSE canonical IR, enabling any downstream SYNAPSE-registered model
    to consume a Docling-processed document without writing custom connector code.

    Supported task type: extract
    Domains: document, general

    Architecture note: This adapter follows the SYNAPSE pure-function contract.
    The caller is responsible for running Docling's DocumentConverter and
    passing the resulting DoclingDocument (or its export_to_dict() equivalent)
    to egress(). The adapter makes no network calls and starts no subprocesses.

    Typical caller usage:
        from docling.document_converter import DocumentConverter
        adapter = DoclingAdapter()
        model_input = adapter.ingress(ir)
        result = DocumentConverter().convert(model_input["source"])
        result_ir = adapter.egress(result.document, ir, latency_ms)
    """

    MODEL_ID = "docling-project/docling"
    ADAPTER_VERSION = "1.0.0"

    # -----------------------------------------------------------------------
    # ingress
    # -----------------------------------------------------------------------

    def ingress(self, ir: CanonicalIR) -> dict[str, Any]:
        """
        Convert canonical IR to Docling DocumentConverter input format.

        Docling's DocumentConverter.convert() accepts a local file path (str)
        or a URL (str) as its source argument. This method extracts the source
        from ir.payload.content and packages it for the caller.

        Args:
            ir: Canonical IR whose payload.content holds a local file path or
                URL pointing to the document to be processed.

        Returns:
            dict with key "source" containing the path or URL string.
            Returns {"source": ""} when payload.content is None or missing.
            Includes "quality_floor" key when ir.task_header.quality_floor is
            set, so callers can use it as a confidence threshold on egress.
        """
        content = ""
        if ir.payload is not None and ir.payload.content is not None:
            content = ir.payload.content

        result: dict[str, Any] = {"source": content}
        if ir.task_header.quality_floor is not None:
            result["quality_floor"] = ir.task_header.quality_floor
        return result

    # -----------------------------------------------------------------------
    # egress
    # -----------------------------------------------------------------------

    def egress(
        self,
        model_output: Any,
        original_ir: CanonicalIR,
        latency_ms: int,
    ) -> CanonicalIR:
        """
        Convert DoclingDocument output to canonical IR.

        Handles both a live DoclingDocument instance (preferred) and the plain
        dict produced by DoclingDocument.export_to_dict() (fallback). Any other
        type returns a safe clone of original_ir with provenance appended and
        no other mutations — this method never raises.

        Duck-typing is used for DoclingDocument detection so the adapter module
        loads cleanly when docling is not installed in the host environment.

        Mapping applied:
          - export_to_markdown()       → payload.content (primary text)
          - doc.texts TextItem list    → payload.entities (one Entity per block)
          - len(doc.tables)            → payload.data["docling_table_count"]
          - len(doc.pages)             → payload.data["docling_page_count"]
          - overall confidence = 1.0   (Docling emits no document-level score)

        Args:
            model_output: DoclingDocument instance or export_to_dict() result.
            original_ir:  The canonical IR passed to ingress().
            latency_ms:   Wall-clock conversion time in milliseconds.

        Returns:
            Updated canonical IR with document content, entities, metadata,
            and provenance appended. original_ir is never mutated.
        """
        try:
            # ------------------------------------------------------------------
            # Input-type detection via duck typing — no docling install required
            # ------------------------------------------------------------------
            is_docling_like = (
                not isinstance(model_output, dict)
                and hasattr(model_output, "export_to_markdown")
                and hasattr(model_output, "texts")
            )
            is_dict = isinstance(model_output, dict)

            if not is_docling_like and not is_dict:
                # Unknown type — return a safe clone with no payload mutations.
                safe = original_ir.clone()
                safe.provenance.append(
                    self.build_provenance(confidence=1.0, latency_ms=latency_ms)
                )
                return safe

            updated = original_ir.clone()

            # ------------------------------------------------------------------
            # Primary text content
            # ------------------------------------------------------------------
            if is_docling_like:
                try:
                    markdown: str = model_output.export_to_markdown()
                except Exception:
                    try:
                        markdown = model_output.export_to_text()
                    except Exception:
                        markdown = ""
                raw_texts: list[Any] = list(model_output.texts or [])
                raw_tables: list[Any] = list(model_output.tables or [])
                raw_pages: dict[Any, Any] = dict(model_output.pages or {})
            else:
                # dict produced by DoclingDocument.export_to_dict()
                text_items: list[Any] = list(model_output.get("texts") or [])
                markdown = "\n\n".join(
                    item.get("text", "")
                    for item in text_items
                    if isinstance(item, dict) and item.get("text")
                )
                raw_texts = text_items
                raw_tables = list(model_output.get("tables") or [])
                raw_pages = dict(model_output.get("pages") or {})

            updated.payload.content = markdown

            # ------------------------------------------------------------------
            # Structured text blocks → entities
            # ------------------------------------------------------------------
            entities: list[Entity] = []
            for item in raw_texts:
                if is_docling_like:
                    text = getattr(item, "text", "") or ""
                    label_attr = getattr(item, "label", None)
                    label = (
                        label_attr.value
                        if label_attr is not None and hasattr(label_attr, "value")
                        else str(label_attr or "")
                    )
                else:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text") or ""
                    label_raw = item.get("label", "")
                    label = label_raw if isinstance(label_raw, str) else str(label_raw)

                if not text:
                    continue

                entities.append(
                    Entity(
                        text=text,
                        label=label,
                        start=None,
                        end=None,
                        confidence=1.0,
                    )
                )

            # Per SYNAPSE spec — use None, never an empty list.
            updated.payload.entities = entities if entities else None

            # ------------------------------------------------------------------
            # Metadata: table and page counts stored in payload.data
            # ------------------------------------------------------------------
            if raw_tables:
                if updated.payload.data is None:
                    updated.payload.data = {}
                updated.payload.data["docling_table_count"] = len(raw_tables)

            if raw_pages:
                if updated.payload.data is None:
                    updated.payload.data = {}
                updated.payload.data["docling_page_count"] = len(raw_pages)

            # ------------------------------------------------------------------
            # Provenance — Docling emits no document-level confidence score
            # ------------------------------------------------------------------
            updated.provenance.append(
                self.build_provenance(confidence=1.0, latency_ms=latency_ms)
            )

            return updated

        except Exception:
            safe = original_ir.clone()
            safe.provenance.append(
                self.build_provenance(confidence=1.0, latency_ms=latency_ms)
            )
            return safe
