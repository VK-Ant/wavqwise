"""Smoke tests for runtime engine."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.mark.smoke
def test_runtime_detection():
    from wavqwise.runtime.engine import RuntimeEngine
    engine = RuntimeEngine()
    assert engine.device is not None
    assert engine.backend is not None


@pytest.mark.smoke
def test_runtime_summary():
    from wavqwise.runtime.engine import RuntimeEngine
    engine = RuntimeEngine()
    summary = engine.summary()
    assert "Backend" in summary
    assert "Device" in summary
    assert "Available" in summary


@pytest.mark.smoke
def test_runtime_is_gpu():
    from wavqwise.runtime.engine import RuntimeEngine
    engine = RuntimeEngine()
    assert isinstance(engine.is_gpu, bool)


@pytest.mark.smoke
def test_runtime_force_cpu():
    from wavqwise.runtime.engine import RuntimeEngine
    engine = RuntimeEngine(preferred="cpu")
    assert engine.device == "cpu"


@pytest.mark.smoke
def test_pipeline_has_runtime():
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    info = p.runtime_info()
    assert "Backend" in info
