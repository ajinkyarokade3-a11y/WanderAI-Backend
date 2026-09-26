"""Shared LLM-based travel preference extraction.

One schema + one system prompt across all providers (Gemini, OpenRouter, Grok).
The router handles provider failover; this module handles schema enforcement,
response validation, and retry-on-malformed-JSON.

Flow:
1. Build an AIRequest with the shared schema and system prompt.
2. Call the failover router (Gemini -> OpenRouter -> Grok).
3. Parse and validate the JSON response.
4. If parsing fails (provider doesn't support strict schema), retry with
   json_object mode (no schema enforcement).
5. If still malformed, raise an AIProviderError.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from backend.ai.ai_types import AIErrorCategory, AIProviderError, AIRequest
from backend.ai.router import generate_with_failover

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared extraction schema — one definition, used by every provider.
# ---------------------------------------------------------------------------

EXTRACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "detected_destination": {"type": ["string", "null"]},
        "detected_origin": {"type": ["string", "null"]},
        "start_date": {"type": ["string", "null"]},
        "end_date": {"type": ["string", "null"]},
        "budget_tier": {
            "type": ["string", "null"],
            "enum": ["budget", "moderate", "luxury", "ultra_luxury", None],
        },
        "budget_amount": {"type": ["number", "null"]},
        "budget_currency": {"type": ["string", "null"]},
        "interests": {"type": "array", "items": {"type": "string"}},
        "travel_companions": {
            "type": ["string", "null"],
            "enum": ["solo", "couple", "family", "friends", None],
        },
        "traveler_count": {"type": ["integer", "null"]},
        "duration_days": {"type": ["integer", "null"]},
        "pace": {
            "type": ["string", "null"],
            "enum": ["relaxed", "balanced", "packed", None],
        },
        "special_requests": {"type": ["string", "null"]},
    },
    "required": [
        "detected_destination",
        "detected_origin",
        "start_date",
        "end_date",
        "budget_tier",
        "budget_amount",
        "budget_currency",
        "interests",
        "travel_companions",
        "traveler_count",
        "duration_days",
        "pace",
        "special_requests",
    ],
    "additionalProperties": False,
}

EXTRACTION_SYSTEM_PROMPT = """\
You are TourFlow AI's Travel Preference Extractor. \
Extract structured travel parameters from the user's message. \
Only extract values that are EXPLICITLY provided by the user. \
Never invent, guess, or use default values. \
If a value is not mentioned, set it to null. \
Return ONLY valid JSON matching the provided schema. \
Do not include markdown fences or any text outside the JSON object."""


def build_extraction_request(
    text_prompt: str,
    context: Optional[Dict[str, Any]] = None,
    *,
    strict: bool = True,
) -> AIRequest:
    """Build an AIRequest for travel preference extraction.

    :param text_prompt: The user's travel request text.
    :param context: Optional context (known destinations, current trip, etc.).
    :param strict: If True, request strict schema enforcement. If False,
        request JSON mode only (for providers that don't support strict schemas).
    """
    context_json = json.dumps(context or {})
    prompt = (
        f"Extract travel preferences from this user message:\n"
        f"\"{text_prompt}\"\n\n"
        f"Context:\n{context_json}"
    )
    return AIRequest(
        prompt=prompt,
        system_instruction=EXTRACTION_SYSTEM_PROMPT,
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=EXTRACTION_SCHEMA if strict else None,
        max_output_tokens=1024,
    )


def parse_extraction_response(text: str) -> Dict[str, Any]:
    """Parse and validate an extraction response.

    :param text: The raw response text from the LLM.
    :raises ValueError: If the response is not valid JSON or not an object.
    """
    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Extraction response must be a JSON object")
    return data


def _validate_extraction(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize an extraction result.

    Ensures all required keys are present and have the correct types.
    """
    result = {}
    for key in EXTRACTION_SCHEMA["required"]:
        result[key] = data.get(key)
    return result


def extract_travel_preferences(
    text_prompt: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract travel preferences using LLM-based structured output.

    Uses the failover router to try Gemini, OpenRouter, and Grok in order.
    First attempts strict schema enforcement; if the response is malformed,
    retries with JSON mode only.

    :param text_prompt: The user's travel request text.
    :param context: Optional context (known destinations, current trip, etc.).
    :raises AIProviderError: If all providers fail or the response is malformed.
    """
    # First attempt: strict schema enforcement
    request = build_extraction_request(text_prompt, context, strict=True)
    try:
        response = generate_with_failover(request)
        data = parse_extraction_response(response.text)
        data["source"] = response.provider
        return _validate_extraction(data)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Strict schema extraction failed (%s), retrying with json_object mode",
            exc,
        )
    except AIProviderError:
        raise  # Provider failure — don't retry, let the caller handle it

    # Retry: JSON mode only (no strict schema)
    request = build_extraction_request(text_prompt, context, strict=False)
    try:
        response = generate_with_failover(request)
        data = parse_extraction_response(response.text)
        data["source"] = response.provider
        return _validate_extraction(data)
    except (json.JSONDecodeError, ValueError) as exc:
        raise AIProviderError(
            f"Extraction failed: malformed JSON response: {exc}",
            category=AIErrorCategory.SERVER_ERROR,
            provider="",
            model="",
        ) from exc
