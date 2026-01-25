"""Unit tests for experiments/04_cloud_sweep/sweep.py Pydantic models."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

# Add experiments to path for import
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "04_cloud_sweep"))

from sweep import (
    SweepConfig,
    TrialRecord,
)


class TestPydanticModels:
    """Test Pydantic model serialization and validation."""

    def test_sweep_config_creation(self) -> None:
        config = SweepConfig(
            model="Qwen/Qwen2.5-3B-Instruct",
            layer=24,
            strength=2.0,
            magnitude=80.0,
            vector_norm=40.0,
            prompt_version="v2",
            inject_style="generation",
            trial=5,
        )
        assert config.model == "Qwen/Qwen2.5-3B-Instruct"
        assert config.layer == 24
        assert config.strength == 2.0

    def test_sweep_config_json_roundtrip(self) -> None:
        config = SweepConfig(
            model="Qwen/Qwen2.5-3B-Instruct",
            layer=24,
            strength=2.0,
            magnitude=80.0,
            vector_norm=40.0,
            prompt_version="v2",
            inject_style="generation",
            trial=5,
        )
        json_str = config.model_dump_json()
        restored = SweepConfig.model_validate_json(json_str)
        assert restored == config

    def test_trial_record_creation(self) -> None:
        config = SweepConfig(
            model="Qwen/Qwen2.5-3B-Instruct",
            layer=24,
            strength=2.0,
            magnitude=80.0,
            vector_norm=40.0,
            prompt_version="v2",
            inject_style="generation",
            trial=5,
        )
        record = TrialRecord(
            id="test_trial_001",
            timestamp="2026-01-23T14:30:22",
            concept="fear",
            was_injected=True,
            response="I notice something unusual...",
            config=config,
        )
        assert record.id == "test_trial_001"
        assert record.was_injected is True

    def test_trial_record_json_roundtrip(self) -> None:
        config = SweepConfig(
            model="Qwen/Qwen2.5-3B-Instruct",
            layer=24,
            strength=2.0,
            magnitude=80.0,
            vector_norm=40.0,
            prompt_version="v2",
            inject_style="generation",
            trial=5,
        )
        record = TrialRecord(
            id="test_trial_001",
            timestamp="2026-01-23T14:30:22",
            concept="ocean",
            was_injected=True,
            response="I sense something about water...",
            config=config,
        )
        json_str = record.model_dump_json()
        restored = TrialRecord.model_validate_json(json_str)
        assert restored == record

    def test_sweep_config_invalid_inject_style_rejected(self) -> None:
        try:
            SweepConfig(
                model="Qwen/Qwen2.5-3B-Instruct",
                layer=24,
                strength=2.0,
                magnitude=80.0,
                vector_norm=40.0,
                prompt_version="v2",
                inject_style="invalid",  # type: ignore[arg-type]
                trial=5,
            )
            assert False, "Should have raised ValidationError"
        except ValidationError:
            pass


class TestTrialRecordOutputFormat:
    """Test that output format matches README specification."""

    def test_output_matches_readme_format(self) -> None:
        """Verify output JSON matches format documented in README."""
        config = SweepConfig(
            model="Qwen/Qwen2.5-3B-Instruct",
            layer=24,
            strength=2.0,
            magnitude=80.0,
            vector_norm=40.0,
            prompt_version="v2",
            inject_style="generation",
            trial=15,
        )
        record = TrialRecord(
            id="fear_injection_L24_S2.0_t15",
            timestamp="2026-01-23T14:30:22",
            concept="fear",
            was_injected=True,
            response="Yes, I notice something unusual...",
            config=config,
        )

        data = record.model_dump()

        # Verify top-level keys
        assert "id" in data
        assert "timestamp" in data
        assert "concept" in data
        assert "was_injected" in data
        assert "response" in data
        assert "config" in data

        # Verify config keys
        assert "model" in data["config"]
        assert "layer" in data["config"]
        assert "strength" in data["config"]
        assert "magnitude" in data["config"]
        assert "vector_norm" in data["config"]
        assert "prompt_version" in data["config"]
        assert "trial" in data["config"]
