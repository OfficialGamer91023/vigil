"""The LLM portfolio manager (PLAN §6): the model *selects*, the kernel *disposes*.

This package is the one place a network call to a language model lives on the
entry path, and it is built so that the model can only ever narrow a set the risk
kernel has already approved. Nothing here can construct a trade, size one, or
reach the broker — it returns an *index into a pre-validated menu*, and even that
choice is re-gated by `execution/router.py` before it binds.

The public surface is deliberately small:

- `PortfolioManager` — wraps the OpenAI Responses API with a hard timeout, strict
  structured output, and a deterministic fallback that is the whole point.
- `Selection` — what `select` returns: the chosen proposal plus the metrics the
  journal needs to answer "did the model actually run, and did its prefix cache?".
- `build_manager` — constructs a manager from `AgentConfig` + the environment, or
  returns `None` when the model is switched off or unkeyed, in which case the
  caller runs the deterministic ranker and never knows the difference.
"""

from __future__ import annotations

from vigil.agent.manager import PortfolioManager, Selection, build_client, build_manager
from vigil.agent.schema import PMSelection

__all__ = ["PortfolioManager", "Selection", "PMSelection", "build_manager", "build_client"]
