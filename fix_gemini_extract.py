from pathlib import Path
import re

file_path = Path("backend/ai/gemini_service.py")

source = file_path.read_text(encoding="utf-8")

start_marker = "    def extract_preferences("
end_marker = "    def recommend("

start_index = source.find(start_marker)
end_index = source.find(end_marker, start_index)

if start_index == -1:
    raise RuntimeError("Could not find extract_preferences() method")

if end_index == -1:
    raise RuntimeError("Could not find recommend() method")

new_method = '''    def extract_preferences(
        self,
        text_prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract structured travel preferences.

        Gemini is used as the primary extractor when available.
        Heuristic extraction fills values that Gemini omits.

        If Gemini fails, the heuristic extractor provides a reliable
        fallback instead of returning incomplete data.
        """

        if not self.is_available() or not self.client:
            result = self._heuristic_preferences(
                text_prompt,
                context
            )
            result["source"] = "fallback_extractor"
            return result

        try:
            prompt = f"""
You are TourFlow AI's Travel Preference Extractor.

Extract only information explicitly provided by the user.

User request:
"{text_prompt}"

Context:
{json.dumps(context or {})}

Rules:
- Do not invent missing values.
- Do not use numbers from examples or placeholders.
- If a value is missing, return null.
- Detect traveler_count from phrases such as "2 people".
- Detect budget_amount and budget_currency when explicitly mentioned.
- Preserve the user's interests.
- Return valid JSON only.

Required JSON schema:

{{
    "detected_destination": null,
    "budget_tier": null,
    "budget_amount": null,
    "budget_currency": null,
    "interests": [],
    "travel_companions": null,
    "traveler_count": null,
    "duration_days": null,
    "pace": null,
    "special_requests": null
}}
"""

            text = None
            used_model = GEMINI_MODEL_FALLBACKS[0]

            for model_name in GEMINI_MODEL_FALLBACKS:
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )

                    response_text = getattr(response, "text", None)

                    if response_text and response_text.strip():
                        text = response_text.strip()
                        used_model = model_name
                        break

                except Exception as exc:
                    logger.warning(
                        "Model %s failed during preference extraction: %s",
                        model_name,
                        exc,
                    )

            if not text:
                raise RuntimeError(
                    "All Gemini models temporarily unavailable"
                )

            if text.startswith("```json"):
                text = text[7:]

            if text.startswith("```"):
                text = text[3:]

            if text.endswith("```"):
                text = text[:-3]

            data = json.loads(text.strip())

            if not isinstance(data, dict):
                raise ValueError(
                    "Gemini preference response must be a JSON object"
                )

            # Run the deterministic extractor as well.
            # This ensures that explicit values omitted by Gemini
            # are still preserved in the API response.
            heuristic = self._heuristic_preferences(
                text_prompt,
                context
            )

            fields_to_merge = (
                "detected_destination",
                "budget_tier",
                "budget_amount",
                "budget_currency",
                "travel_companions",
                "traveler_count",
                "duration_days",
                "pace",
                "special_requests",
            )

            for key in fields_to_merge:
                gemini_value = data.get(key)
                heuristic_value = heuristic.get(key)

                if (
                    gemini_value is None
                    and heuristic_value is not None
                ):
                    data[key] = heuristic_value

            # Merge interests from both sources without duplicates.
            gemini_interests = data.get("interests") or []
            heuristic_interests = heuristic.get("interests") or []

            if not isinstance(gemini_interests, list):
                gemini_interests = [str(gemini_interests)]

            if not isinstance(heuristic_interests, list):
                heuristic_interests = [str(heuristic_interests)]

            merged_interests = list(
                dict.fromkeys(
                    [
                        *gemini_interests,
                        *heuristic_interests,
                    ]
                )
            )

            data["interests"] = merged_interests
            data["source"] = used_model

            return data

        except Exception as exc:
            logger.error(
                "Gemini preference extraction error: %s",
                exc,
            )

            # Always return heuristic values when Gemini fails.
            result = self._heuristic_preferences(
                text_prompt,
                context
            )

            result["error"] = str(exc)[:200]
            result["source"] = "fallback_on_error"

            return result

'''

updated_source = (
    source[:start_index]
    + new_method
    + source[end_index:]
)

file_path.write_text(updated_source, encoding="utf-8")

print("SUCCESS: extract_preferences() was replaced.")
print(f"Updated file: {file_path}")