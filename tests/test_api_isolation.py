"""Hard rule #6, enforced structurally: **the API service never trades.**

`src/vigil/api/` may read the journal and write control flags. It must never
place an order, hold trading state, or sit between the risk kernel and the
broker — and the worker must run correctly with the API, the web app and Redis
all stopped.

A docstring saying so is a hope. This walks the import graph of every module in
`vigil.api` transitively and fails if any of them can reach the broker, the
submit path or the kernel. Static analysis rather than a runtime probe, because
the violation being pinned is *reachability*, not whether a particular request
happened to call it: `from alpaca.trading.client import TradingClient` inside a
route nobody exercised in tests is exactly as much a violation, and a runtime
test would pass right over it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

# Reaching any of these from `vigil.api` means the rule is broken. `vigil.risk` is
# on the list not because the kernel is dangerous — it is pure and has no network
# — but because an API that evaluates gates has started to hold trading state, and
# §2.2 says the kernel's answers belong to the worker alone.
FORBIDDEN = (
    "alpaca",
    "vigil.data.alpaca_client",
    "vigil.data.chain",
    "vigil.execution",
    "vigil.risk",
    "vigil.strategy",
    "vigil.worker",
)


def module_name(path: pathlib.Path) -> str:
    return ".".join(path.relative_to(SRC).with_suffix("").parts).removesuffix(".__init__")


def imports_of(path: pathlib.Path) -> set[str]:
    """Every module this file imports, `import x` and `from x import y` alike.

    Relative imports are resolved against the file's own package, so a future
    `from .deps import Db` is followed rather than silently skipped — an unfollowed
    edge is a hole in the check.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    name = module_name(path)
    # `__init__.py` *is* its package; any other module lives one level under it.
    package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                found.add(f"{base}.{node.module}" if node.module else base)
            elif node.module:
                found.add(node.module)
    return found


def reachable_from(roots: list[pathlib.Path]) -> dict[str, str]:
    """Transitive closure over first-party modules. Maps module -> who imported it.

    The provenance is carried so a failure names the *path* into the forbidden
    module rather than only its name — "api.deps imported vigil.worker.broker" is
    actionable; "something imports the broker" is a scavenger hunt.

    **`parsed` and `provenance` are separate on purpose**, and the first version of
    this function conflated them. Recording a module as seen at the moment an edge
    pointed at it marked it visited *before* it had been opened, so a root file
    that another root imported first was never parsed at all — and an `import`
    added to `deps.py` sailed straight through a green test. A visited-set that
    means "we have an edge to this" cannot also mean "we have read this."
    """
    provenance: dict[str, str] = {}
    parsed: set[str] = set()
    stack = [(module_name(p), "<root>", p) for p in roots]

    while stack:
        name, via, path = stack.pop()
        provenance.setdefault(name, via)
        if name in parsed:
            continue
        parsed.add(name)

        for target in imports_of(path):
            provenance.setdefault(target, name)
            if not target.startswith("vigil.") or target in parsed:
                continue
            # Follow first-party edges into their files. A `from vigil.x import y`
            # where y is a symbol, not a module, simply has no file and stops here.
            for candidate in (
                SRC / (target.replace(".", "/") + ".py"),
                SRC / target.replace(".", "/") / "__init__.py",
            ):
                if candidate.exists():
                    stack.append((target, name, candidate))
                    break
    return provenance


@pytest.fixture(scope="module")
def graph() -> dict[str, str]:
    api = SRC / "vigil" / "api"
    assert api.is_dir(), "src/vigil/api/ is missing"
    return reachable_from(sorted(api.rglob("*.py")))


@pytest.mark.parametrize("forbidden", FORBIDDEN)
def test_the_api_cannot_reach_the_trading_path(graph: dict[str, str], forbidden: str) -> None:
    hits = {
        module: via
        for module, via in graph.items()
        if module == forbidden or module.startswith(forbidden + ".")
    }
    assert not hits, (
        f"`vigil.api` can reach `{forbidden}` — hard rule #6 says the API service "
        f"never trades. Reached via: "
        + "; ".join(f"{via} -> {module}" for module, via in sorted(hits.items()))
    )


def test_only_deps_touches_settings_and_only_for_the_env_path() -> None:
    """The API never constructs a client, so it has no reason to read Alpaca keys.

    Checked against the package's own files rather than the transitive graph:
    `vigil.settings` is legitimately reachable *through* `vigil.db.session`, and a
    graph assertion would depend on which edge happened to be walked first. What
    actually matters is narrower and exactly checkable — no module in `vigil.api`
    imports it except `deps.py`, and that one takes `REPO_ROOT` only, to locate
    `.env` for the control token.

    The consequence is worth stating: an API container with an Alpaca-free `.env`
    still serves the journal.
    """
    offenders = {
        module_name(f): sorted(i for i in imports_of(f) if i.startswith("vigil.settings"))
        for f in sorted((SRC / "vigil" / "api").rglob("*.py"))
        if any(i.startswith("vigil.settings") for i in imports_of(f))
    }
    assert set(offenders) <= {"vigil.api.deps"}, offenders

    # The *symbols* it takes, from the AST rather than a substring search — the
    # docstring below names `load_settings` while explaining why it is not called,
    # and a grep cannot tell an explanation from a call.
    deps = SRC / "vigil" / "api" / "deps.py"
    tree = ast.parse(deps.read_text(), filename=str(deps))
    symbols = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("vigil.settings")
        for alias in node.names
    }
    assert symbols == {"REPO_ROOT"}, (
        f"deps.py takes {sorted(symbols)} from vigil.settings. Only REPO_ROOT is "
        f"justified — `load_settings()` asserts paper mode and demands Alpaca "
        f"credentials the API has no business holding."
    )


def test_the_shared_control_vocabulary_imports_nothing(graph: dict[str, str]) -> None:
    """`vigil.control` is the whole channel between the API and the worker.

    Two strings and a frozenset. If it grows an import, the two processes have
    started to share more than a vocabulary, and the next thing they share is
    state.
    """
    assert imports_of(SRC / "vigil" / "control.py") <= {"__future__", "typing"}
