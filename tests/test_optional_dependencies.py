import pytest

from alprg.engines.fake_engine import FakeEngine
from alprg.engines.fast_alpr_engine import FastALPREngine
from alprg.engines.openalpr_engine import OpenALPREngine
from alprg.harness import MultiEngineHarness


def test_importing_optional_engines_never_raises() -> None:
    """Importing these modules must not raise even when fast_alpr/openalpr
    aren't installed -- already proven by this test file importing them above,
    but assert explicitly that the availability flags reflect that."""
    assert FastALPREngine.is_available() is False
    assert OpenALPREngine.is_available() is False


def test_fast_alpr_load_raises_actionable_runtime_error() -> None:
    engine = FastALPREngine()
    with pytest.raises(RuntimeError, match="fast_alpr is not installed"):
        engine.load()


def test_openalpr_load_raises_actionable_runtime_error() -> None:
    engine = OpenALPREngine()
    with pytest.raises(RuntimeError, match="openalpr Python bindings not found"):
        engine.load()


def test_harness_never_hard_requires_gpu_engines() -> None:
    """The harness itself must work fine when explicitly given only FakeEngine,
    regardless of which real engines are/aren't installed on this machine."""
    with MultiEngineHarness(engines=[FakeEngine()]) as harness:
        assert harness.engine_names == ["fake"]
