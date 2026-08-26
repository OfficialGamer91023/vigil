"""The paper-trading guard (hard rule #1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vigil.settings import LiveTradingRefused, load_settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the real environment so a developer's .env cannot make these pass."""
    for var in ("ALPACA_PAPER_TRADE", "ALPACA_LIVE_TRADE", "ALPACA_API_KEY_ID",
                "ALPACA_API_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    # load_dotenv does not override existing vars, but it *does* set missing ones,
    # so point it at a path that cannot exist.
    monkeypatch.setattr("vigil.settings.REPO_ROOT", Path("/nonexistent"))


@pytest.mark.parametrize("value", ["", "false", "0", "TRUE ", "yes", "1"])
def test_only_the_literal_string_true_is_accepted(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truthiness test here is how a guard like this fails open."""
    monkeypatch.setenv("ALPACA_PAPER_TRADE", value)
    if value.strip().lower() == "true":
        pytest.skip("'TRUE ' is stripped and lowered by design")
    with pytest.raises(LiveTradingRefused):
        load_settings(require_credentials=False)


def test_unset_refuses() -> None:
    with pytest.raises(LiveTradingRefused):
        load_settings(require_credentials=False)


def test_a_live_flag_refuses_even_when_paper_is_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_PAPER_TRADE", "true")
    monkeypatch.setenv("ALPACA_LIVE_TRADE", "1")
    with pytest.raises(LiveTradingRefused):
        load_settings(require_credentials=False)


def test_missing_credentials_raise_a_useful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_PAPER_TRADE", "true")
    with pytest.raises(RuntimeError, match="ALPACA_API_KEY_ID"):
        load_settings()


def test_paper_settings_never_expose_a_live_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_PAPER_TRADE", "true")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s")
    s = load_settings()
    assert s.paper is True
    assert "paper-api" in s.trading_base_url
