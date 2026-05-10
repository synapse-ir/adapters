import type OpenAI from "openai";
import { SynapseAdapter } from "../base.js";
import type { CanonicalIR } from "../types.js";

// GPT-4o-mini pricing (USD per token, as of 2025-05)
const PRICE_INPUT = 0.15 / 1_000_000;
const PRICE_OUTPUT = 0.6 / 1_000_000;

const MODEL_NAME = "gpt-4o-mini";

const DEFAULT_SYSTEM_PROMPT =
  "You are a zero-shot text classifier. Classify the input into exactly one of the " +
  "provided labels. Respond with valid JSON only, schema: " +
  '{"label": "<chosen label>", "confidence": <float 0.0–1.0>}';

/** Payload sent to the OpenAI chat completions endpoint. */
export interface ClauseExtractorInput {
  model: string;
  messages: Array<{ role: "system" | "user"; content: string }>;
  response_format: { type: "json_object" };
}

/** Raw response received from the OpenAI chat completions endpoint. */
export type ClauseExtractorOutput = OpenAI.Chat.ChatCompletion;

/**
 * Zero-shot classifier adapter backed by GPT-4o-mini.
 *
 * Ingress builds a JSON-mode chat request; egress parses the structured
 * `{label, confidence}` response and writes it into `payload.data`.
 * Cost is estimated from token usage at current gpt-4o-mini list pricing.
 */
export class OpenAIClassifierAdapter extends SynapseAdapter<
  ClauseExtractorInput,
  ClauseExtractorOutput
> {
  readonly modelId = "openai/gpt-4o-mini-classifier";
  readonly adapterVersion = "1.0.0";

  constructor(
    readonly client: OpenAI,
    private readonly labels: string[],
    private readonly systemPrompt?: string,
  ) {
    super();
  }

  ingress(ir: CanonicalIR): ClauseExtractorInput {
    const sys = this.systemPrompt ?? DEFAULT_SYSTEM_PROMPT;
    const labelList = this.labels.join(", ");
    return {
      model: MODEL_NAME,
      messages: [
        {
          role: "system",
          content: `${sys}\n\nLabels: [${labelList}]`,
        },
        {
          role: "user",
          content: ir.payload.content ?? "",
        },
      ],
      response_format: { type: "json_object" },
    };
  }

  egress(
    output: ClauseExtractorOutput,
    originalIr: CanonicalIR,
    latencyMs: number,
  ): CanonicalIR {
    const updated = originalIr.clone();
    const raw = output.choices[0]?.message.content ?? "";

    let label = "";
    let confidence = 0.0;
    let warning: string | undefined;

    try {
      const parsed = JSON.parse(raw) as Record<string, unknown>;
      if (typeof parsed["label"] === "string" && typeof parsed["confidence"] === "number") {
        label = parsed["label"];
        confidence = Math.max(0, Math.min(1, parsed["confidence"]));
      } else {
        warning = `Malformed response: missing or invalid fields in: ${raw}`;
      }
    } catch {
      warning = `Malformed JSON from model: ${raw}`;
    }

    const usage = output.usage;
    const tokenCount = usage?.total_tokens ?? 0;
    const costUsd = usage
      ? usage.prompt_tokens * PRICE_INPUT + usage.completion_tokens * PRICE_OUTPUT
      : 0;

    updated.payload.data = { label, confidence, token_count: tokenCount };

    // G-S04: propagate pii_present flag from input IR
    if (originalIr.payload["pii_present"] === true) {
      updated.payload["pii_present"] = true;
    }

    updated.provenance.push(
      this.buildProvenance(confidence, latencyMs, {
        costUsd,
        ...(warning ? { metadata: { warning } } : {}),
      }),
    );

    return updated;
  }
}
