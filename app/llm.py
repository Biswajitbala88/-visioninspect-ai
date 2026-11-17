"""GenAI layer: turn raw CV output into a QA inspection report with Claude (vision + structured output)."""
from __future__ import annotations

import json
import os

import anthropic

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "disposition": {"type": "string", "enum": ["accept", "rework", "scrap", "manual_review"]},
        "severity": {"type": "string", "enum": ["none", "minor", "major", "critical"]},
        "summary": {"type": "string"},
        "observations": {"type": "array", "items": {"type": "string"}},
        "likely_root_causes": {"type": "array", "items": {"type": "string"}},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "model_agreement": {"type": "string",
                            "description": "Whether your visual read of the image agrees with the CV model, and why."},
    },
    "required": ["disposition", "severity", "summary", "observations", "likely_root_causes",
                 "recommended_actions", "model_agreement"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are a quality-assurance engineer on a metal-surface production line. You receive a "
    "grayscale surface image, the output of an in-house CNN defect classifier with Grad-CAM "
    "regions, and optional operator notes. Write a concise inspection report. Check the CNN's "
    "claim against what you can see in the image; if they disagree or confidence is low, choose "
    "manual_review. Keep each list item to one sentence."
)


class LLMUnavailable(RuntimeError):
    pass


def generate_report(image_png_b64: str, inspection: dict, notes: str = "") -> dict:
    try:
        client = anthropic.Anthropic()
    except anthropic.AnthropicError as e:  # no credentials configured
        raise LLMUnavailable("Set ANTHROPIC_API_KEY to enable GenAI reports.") from e

    facts = {k: inspection[k] for k in ("label", "confidence", "probabilities", "regions") if k in inspection}
    data = image_png_b64.split(",", 1)[-1]
    try:
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": REPORT_SCHEMA}},
            system=SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                {"type": "text", "text": f"CNN output:\n{json.dumps(facts, indent=2)}\n\n"
                                         f"Operator notes: {notes or '(none)'}"},
            ]}],
        )
    except TypeError as e:  # SDK raises this at request time when no credential source resolves
        if "authentication" not in str(e).lower():
            raise
        raise LLMUnavailable("Set ANTHROPIC_API_KEY to enable GenAI reports.") from e
    except anthropic.AuthenticationError as e:
        raise LLMUnavailable("Anthropic credentials were rejected.") from e
    except anthropic.APIConnectionError as e:
        raise LLMUnavailable("Could not reach the Anthropic API.") from e

    if resp.stop_reason == "refusal":
        raise LLMUnavailable("The model declined to produce a report for this input.")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    report = json.loads(text)
    report["_meta"] = {"model": resp.model, "input_tokens": resp.usage.input_tokens,
                       "output_tokens": resp.usage.output_tokens}
    return report
