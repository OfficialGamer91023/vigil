"""The model's output contract — flat on purpose, and hardened for strict mode.

**Why flat.** OpenAI's structured-output *strict* mode requires that every object
in the schema set `additionalProperties: false` and mark **every** property
`required` (optionality is expressed with a nullable type, not by omission). Nested
Pydantic models multiply the places that has to hold and are the usual source of a
`400 invalid schema`. A single flat object with four scalar fields has exactly one
object to harden, so `strict: true` is enforceable without fighting the serializer.

**Why so little.** The kernel has already validated everything tradeable. The model
is not describing a trade — it is answering "which of these, if any, and why". So
the schema carries an index, a stand-down flag, a confidence, and a sentence, and
nothing that could smuggle an unvalidated number toward the broker.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PMSelection(BaseModel):
    """One portfolio-manager decision over a menu of pre-approved candidates."""

    stand_down: bool = Field(
        description=(
            "True to decline every candidate this cycle. A valid, common answer: "
            "the menu is legal but nothing on it is worth the risk right now."
        )
    )
    choice: int = Field(
        description=(
            "Zero-based index of the chosen candidate in the menu. Ignored when "
            "stand_down is true; set it to 0 in that case. An out-of-range value "
            "is treated as a malformed answer and the deterministic ranker decides."
        )
    )
    confidence: float = Field(
        description=(
            "Conviction in [0,1]. Per §6.1 confidence may only ever *reduce* "
            "conviction-scaled behaviour; it never enlarges a position, because "
            "size was fixed by the kernel before the model saw the menu."
        )
    )
    memo: str = Field(
        description=(
            "One or two sentences of human-readable rationale, stored with the "
            "trade. Text only — it has no effect on execution."
        )
    )


def strict_json_schema(model: type[BaseModel], name: str) -> dict[str, Any]:
    """A `text.format` block for the Responses API, hardened for `strict: true`.

    Pydantic's `model_json_schema()` does not, on its own, satisfy strict mode:
    it omits `additionalProperties: false` and lists only genuinely-required keys
    in `required`. This walks every object node and sets both, which is the whole
    difference between a schema the API accepts and a 400. Kept as a function
    rather than hand-written JSON so the schema cannot drift from the Pydantic
    model that parses the response — one source of truth for the contract.
    """
    schema = model.model_json_schema()

    def _harden(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            props = node.get("properties", {})
            # Strict mode wants *all* properties required; nullability, not
            # omission, is how the model declines to fill one.
            node["required"] = list(props.keys())
            for child in props.values():
                _harden(child)
        # Recurse through the containers a schema can nest objects inside.
        for key in ("items", "anyOf", "allOf", "oneOf"):
            sub = node.get(key)
            if isinstance(sub, list):
                for item in sub:
                    _harden(item)
            elif isinstance(sub, dict):
                _harden(sub)
        for sub in node.get("$defs", {}).values():
            _harden(sub)

    _harden(schema)
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": True,
        }
    }
