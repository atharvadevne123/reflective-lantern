"""Tests for aggregate_signals() consensus voting utility."""

from __future__ import annotations

import pytest

from app.signals import SignalStrength, TradingSignal, aggregate_signals, classify_signal


def _make_signal(direction: str, confidence: float = 0.8) -> TradingSignal:
    pred = 1 if direction == "BUY" else 0
    return classify_signal(prediction=pred, confidence=confidence, horizon_days=5)


class TestAggregateSignals:
    def test_all_buy_gives_buy_consensus(self) -> None:
        signals = [_make_signal("BUY") for _ in range(3)]
        result = aggregate_signals(signals)
        assert result["consensus"] == "BUY"
        assert result["buy_votes"] == 3

    def test_all_sell_gives_sell_consensus(self) -> None:
        signals = [_make_signal("SELL") for _ in range(3)]
        result = aggregate_signals(signals)
        assert result["consensus"] == "SELL"
        assert result["sell_votes"] == 3

    def test_tie_gives_neutral(self) -> None:
        signals = [_make_signal("BUY"), _make_signal("SELL")]
        result = aggregate_signals(signals)
        assert result["consensus"] == "NEUTRAL"

    def test_empty_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            aggregate_signals([])

    def test_mean_confidence_correct(self) -> None:
        s1 = _make_signal("BUY", confidence=0.8)
        s2 = _make_signal("BUY", confidence=0.6)
        result = aggregate_signals([s1, s2])
        assert abs(result["mean_confidence"] - 0.70) < 1e-3

    def test_single_signal_buy(self) -> None:
        result = aggregate_signals([_make_signal("BUY")])
        assert result["buy_votes"] == 1
        assert result["sell_votes"] == 0
        assert result["consensus"] == "BUY"

    def test_single_signal_sell(self) -> None:
        result = aggregate_signals([_make_signal("SELL")])
        assert result["consensus"] == "SELL"

    def test_vote_counts_sum_to_total(self) -> None:
        signals = [_make_signal("BUY"), _make_signal("BUY"), _make_signal("SELL")]
        result = aggregate_signals(signals)
        assert result["buy_votes"] + result["sell_votes"] == 3

    @pytest.mark.parametrize("n_buy,n_sell,expected", [
        (5, 2, "BUY"),
        (2, 5, "SELL"),
        (3, 3, "NEUTRAL"),
        (1, 0, "BUY"),
    ])
    def test_consensus_parametrized(self, n_buy: int, n_sell: int, expected: str) -> None:
        signals = [_make_signal("BUY") for _ in range(n_buy)] + \
                  [_make_signal("SELL") for _ in range(n_sell)]
        result = aggregate_signals(signals)
        assert result["consensus"] == expected
