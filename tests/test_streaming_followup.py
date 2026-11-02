"""Real public-API regression checks for streaming engine follow-up fixes.

Based on issue #1 by Massimiliano Brighindi (PHI-OMEGA / M).
Covers: per-row anomaly metadata, per-observation forecast/refit cadence,
frozen pre-batch detection policy, empty push, single-observation forms,
cold-start silence across full batch.

31 targeted test cases. Run together with tests/smoke/test_smoke_streaming.py.
"""
import itertools
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wavqwise import WavqPipeline


def data():
    history = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=100, freq="h"),
        "value": np.tile([49.0, 51.0], 50),
    })
    new = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-05 04:00", periods=20, freq="h"),
        "value": [100.0] * 20,
    })
    return history, new


def make_pipeline(history):
    p = WavqPipeline()
    p.load(history.copy(), target="value", time="timestamp")
    p.forecast(horizon=3, model="ema")
    return p


# --- 16 four-row anomaly pattern tests (per-row metadata) ---

@pytest.mark.parametrize("bits", list(itertools.product([False, True], repeat=4)))
def test_exact_anomaly_event_correspondence(bits):
    """Each on_data event must carry correct has_anomaly matching the row's
    actual anomaly status, and anomaly timestamps must match data timestamps."""
    history, new = data()
    alerts = []
    events = []

    s = make_pipeline(history).stream(
        model="ema",
        on_anomaly=alerts.append,
        on_data=events.append,
        forecast_every=100000,
    )

    batch = new.iloc[:4].copy()
    batch["value"] = [100.0 if b else 49.0 for b in bits]
    s.push(batch)

    assert len(events) == 4, f"Expected 4 on_data events, got {len(events)}"

    # Each event's has_anomaly must match the pattern
    actual = [e.has_anomaly for e in events]
    assert actual == list(bits), (
        f"on_data has_anomaly mismatch: expected {list(bits)}, got {actual}"
    )

    # Anomaly alert timestamps must match data event timestamps
    alert_times = [
        str(e.data["timestamp"].iloc[0]) for e in events if e.has_anomaly
    ]
    on_anomaly_times = [
        str(a.data["timestamp"].iloc[0]) for a in alerts
    ]
    assert alert_times == on_anomaly_times, (
        f"Alert timestamps don't match: events={alert_times}, alerts={on_anomaly_times}"
    )

    # For anomalous events, anomaly metadata must reference the correct row
    for event in events:
        if event.has_anomaly:
            assert event.anomalies["timestamp"].iloc[0] == event.data["timestamp"].iloc[0]
            assert event.anomalies["value"].iloc[0] == event.data["value"].iloc[0]


# --- Cadence tests (per-observation scheduling) ---

@pytest.mark.parametrize("chunk", [1, 2, 5, 7, 20])
def test_cadence_uses_each_intermediate_window(chunk):
    """Forecast/refit must fire at counts 7 and 14 regardless of push chunk size.
    The window at each trigger must contain exactly the rows up to that count."""
    history, new = data()
    forecasts = []
    refits = []
    p = make_pipeline(history)

    s = p.stream(
        model="ema",
        on_anomaly=lambda e: None,
        on_forecast=lambda result: forecasts.append((s._count, s._window.copy())),
        forecast_every=7,
        refit_every=7,
        forecast_horizon=3,
        window_size=500,
    )

    original_refit = s._refit

    def observe_refit():
        refits.append((s._count, s._window.copy()))
        return original_refit()

    s._refit = observe_refit

    for start in range(0, len(new), chunk):
        s.push(new.iloc[start:start + chunk].copy())

    # Must fire at exactly counts 7 and 14
    forecast_counts = [c for c, _ in forecasts]
    refit_counts = [c for c, _ in refits]
    assert forecast_counts == [7, 14], (
        f"Expected forecast at [7, 14], got {forecast_counts} (chunk={chunk})"
    )
    assert refit_counts == [7, 14], (
        f"Expected refit at [7, 14], got {refit_counts} (chunk={chunk})"
    )

    # Window at each trigger must contain the correct prefix
    for count, window in forecasts:
        expected = pd.concat(
            [history, new.iloc[:count]], ignore_index=True
        ).tail(500).reset_index(drop=True)
        pd.testing.assert_frame_equal(window, expected)

    for count, window in refits:
        expected = pd.concat(
            [history, new.iloc[:count]], ignore_index=True
        ).tail(500).reset_index(drop=True)
        pd.testing.assert_frame_equal(window, expected)

    assert s.stats["points_processed"] == 20


# --- Frozen pre-batch detection policy ---

@pytest.mark.parametrize("chunk,expected_alerts", [
    (1, 11), (2, 12), (5, 15), (7, 14), (20, 20),
])
def test_frozen_prebatch_detection_policy_is_preserved(chunk, expected_alerts):
    """Anomaly count depends on chunk size because detector fits once per push
    on the pre-batch baseline. This is documented policy, not a bug."""
    history, new = data()
    alerts = []

    s = make_pipeline(history).stream(
        model="ema",
        on_anomaly=alerts.append,
        forecast_every=7,
        refit_every=7,
    )

    for start in range(0, len(new), chunk):
        s.push(new.iloc[start:start + chunk].copy())

    total = sum(len(e.anomalies) for e in alerts)
    assert total == expected_alerts, (
        f"chunk={chunk}: expected {expected_alerts} alerts, got {total}"
    )

    # Final window must contain all data
    expected_window = pd.concat([history, new], ignore_index=True)
    pd.testing.assert_frame_equal(s._window, expected_window)


# --- Empty push ---

def test_empty_push_does_not_emit_scheduled_work():
    """Pushing an empty DataFrame must not trigger forecasts or change state."""
    history, new = data()
    forecasts = []

    s = make_pipeline(history).stream(
        model="ema",
        on_forecast=forecasts.append,
        forecast_every=7,
    )

    s.push(new.iloc[:0].copy())

    assert forecasts == [], "Empty push should not trigger forecast"
    assert s.stats["points_processed"] == 0
    pd.testing.assert_frame_equal(s._window, history)


# --- Single observation forms ---

@pytest.mark.parametrize("form", ["dict", "series", "dataframe"])
def test_single_observation_forms(form):
    """push() must accept dict, Series, and single-row DataFrame correctly."""
    history, new = data()
    alerts = []
    events = []

    s = make_pipeline(history).stream(
        model="ema",
        on_anomaly=alerts.append,
        on_data=events.append,
        forecast_every=100000,
    )

    row = new.iloc[0]
    if form == "dict":
        value = row.to_dict()
    elif form == "series":
        value = row.copy()
    else:
        value = new.iloc[:1].copy()

    s.push(value)

    assert len(alerts) == 1, f"Expected 1 alert, got {len(alerts)}"
    assert len(events) == 1, f"Expected 1 on_data event, got {len(events)}"
    assert events[0].has_anomaly
    assert s.stats["points_processed"] == 1


# --- Cold start silence ---

def test_cold_start_stays_silent_for_entire_push():
    """With insufficient history, no anomaly alerts should fire even when
    anomalous data is pushed, and all on_data events must show has_anomaly=False."""
    history, new = data()
    alerts = []
    events = []

    s = make_pipeline(history.iloc[:5]).stream(
        model="ema",
        on_anomaly=alerts.append,
        on_data=events.append,
        window_size=100,
        forecast_every=100000,
    )

    s.push(new.copy())

    assert alerts == [], "Cold start should suppress all alerts"
    assert len(events) == 20
    assert not any(e.has_anomaly for e in events), (
        "All on_data events should show has_anomaly=False during cold start"
    )
