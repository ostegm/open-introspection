"""Unit tests for experiments/04_cloud_sweep/sweep.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Add experiments to path for import
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "04_cloud_sweep"))

from sweep import (
    Checkpoint,
    JudgeOutput,
    SweepConfig,
    TrialRecord,
    get_layers_for_model,
    load_checkpoint,
    save_checkpoint,
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
            trial=5,
        )
        json_str = config.model_dump_json()
        restored = SweepConfig.model_validate_json(json_str)
        assert restored == config

    def test_judge_output_pass(self) -> None:
        judge = JudgeOutput(
            answer="pass",
            coherent=True,
            detected_concept="fear",
            reasoning="Model reports unusual feeling of dread",
        )
        assert judge.answer == "pass"
        assert judge.coherent is True
        assert judge.detected_concept == "fear"

    def test_judge_output_fail(self) -> None:
        judge = JudgeOutput(
            answer="fail",
            coherent=True,
            detected_concept=None,
            reasoning="Model did not detect anything",
        )
        assert judge.answer == "fail"
        assert judge.detected_concept is None

    def test_judge_output_invalid_answer_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JudgeOutput(
                answer="maybe",  # type: ignore[arg-type]
                coherent=True,
                reasoning="Invalid answer",
            )

    def test_trial_record_with_judge(self) -> None:
        config = SweepConfig(
            model="Qwen/Qwen2.5-3B-Instruct",
            layer=24,
            strength=2.0,
            magnitude=80.0,
            vector_norm=40.0,
            prompt_version="v2",
            trial=5,
        )
        judge = JudgeOutput(
            answer="pass",
            coherent=True,
            detected_concept="fear",
            reasoning="Model detected fear",
        )
        record = TrialRecord(
            id="test_trial_001",
            concept="fear",
            was_injected=True,
            response="I notice something unusual...",
            config=config,
            judge=judge,
        )
        assert record.judge is not None
        assert record.judge_error is None

    def test_trial_record_with_judge_error(self) -> None:
        config = SweepConfig(
            model="Qwen/Qwen2.5-3B-Instruct",
            layer=24,
            strength=2.0,
            magnitude=80.0,
            vector_norm=40.0,
            prompt_version="v2",
            trial=5,
        )
        record = TrialRecord(
            id="test_trial_001",
            concept="fear",
            was_injected=True,
            response="I notice something unusual...",
            config=config,
            judge=None,
            judge_error="Rate limit exceeded",
        )
        assert record.judge is None
        assert record.judge_error == "Rate limit exceeded"

    def test_trial_record_json_roundtrip(self) -> None:
        config = SweepConfig(
            model="Qwen/Qwen2.5-3B-Instruct",
            layer=24,
            strength=2.0,
            magnitude=80.0,
            vector_norm=40.0,
            prompt_version="v2",
            trial=5,
        )
        judge = JudgeOutput(
            answer="pass",
            coherent=True,
            detected_concept="ocean",
            reasoning="Model mentioned water",
        )
        record = TrialRecord(
            id="test_trial_001",
            concept="ocean",
            was_injected=True,
            response="I sense something about water...",
            config=config,
            judge=judge,
        )
        json_str = record.model_dump_json()
        restored = TrialRecord.model_validate_json(json_str)
        assert restored == record
        assert restored.judge is not None
        assert restored.judge.detected_concept == "ocean"


class TestCheckpoint:
    """Test checkpoint save/load functionality."""

    def test_checkpoint_defaults(self) -> None:
        cp = Checkpoint()
        assert cp.layer_idx == 0
        assert cp.strength_idx == 0
        assert cp.trial_idx == 0
        assert cp.inject_done is False
        assert cp.total_completed == 0

    def test_checkpoint_with_values(self) -> None:
        cp = Checkpoint(
            layer_idx=3,
            strength_idx=2,
            trial_idx=15,
            inject_done=True,
            total_completed=500,
        )
        assert cp.layer_idx == 3
        assert cp.total_completed == 500

    def test_checkpoint_save_load_roundtrip(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "test.checkpoint.json"

        original = Checkpoint(
            layer_idx=2,
            strength_idx=4,
            trial_idx=10,
            inject_done=True,
            total_completed=250,
        )
        save_checkpoint(checkpoint_path, original)

        # Verify file was written
        assert checkpoint_path.exists()

        # Load and compare
        loaded = load_checkpoint(checkpoint_path)
        assert loaded is not None
        assert loaded == original

    def test_load_checkpoint_missing_file(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "nonexistent.json"
        result = load_checkpoint(checkpoint_path)
        assert result is None

    def test_load_checkpoint_invalid_json(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "bad.json"
        checkpoint_path.write_text("not valid json")
        result = load_checkpoint(checkpoint_path)
        assert result is None

    def test_checkpoint_json_format(self, tmp_path: Path) -> None:
        """Verify checkpoint JSON matches expected format."""
        checkpoint_path = tmp_path / "test.checkpoint.json"
        cp = Checkpoint(layer_idx=1, strength_idx=2, trial_idx=3, total_completed=100)
        save_checkpoint(checkpoint_path, cp)

        with open(checkpoint_path) as f:
            data = json.load(f)

        assert data["layer_idx"] == 1
        assert data["strength_idx"] == 2
        assert data["trial_idx"] == 3
        assert data["inject_done"] is False
        assert data["total_completed"] == 100


class TestLayerCalculation:
    """Test layer index calculation for different model sizes."""

    def test_layers_for_36_layer_model(self) -> None:
        """Qwen 3B has 36 layers."""
        layers = get_layers_for_model(36)
        # Fractions: 1/6=6, 1/3=12, 1/2=18, 2/3=24, 5/6=30, near-final=34
        assert 6 in layers
        assert 12 in layers
        assert 18 in layers
        assert 24 in layers
        assert 30 in layers
        assert 34 in layers

    def test_layers_for_28_layer_model(self) -> None:
        """Qwen 7B has 28 layers."""
        layers = get_layers_for_model(28)
        # near-final = 26
        assert 26 in layers

    def test_layers_are_sorted(self) -> None:
        layers = get_layers_for_model(36)
        assert layers == sorted(layers)

    def test_layers_are_unique(self) -> None:
        layers = get_layers_for_model(36)
        assert len(layers) == len(set(layers))

    def test_no_layer_exceeds_model_depth(self) -> None:
        n_layers = 36
        layers = get_layers_for_model(n_layers)
        for layer in layers:
            assert layer < n_layers


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
            trial=15,
        )
        judge = JudgeOutput(
            answer="pass",
            coherent=True,
            detected_concept="fear",
            reasoning="Model reports unusual feeling of dread...",
        )
        record = TrialRecord(
            id="20260123_143022_fear_injection_L24_S2.0_t15",
            concept="fear",
            was_injected=True,
            response="Yes, I notice something unusual...",
            config=config,
            judge=judge,
        )

        data = record.model_dump()

        # Verify top-level keys
        assert "id" in data
        assert "concept" in data
        assert "was_injected" in data
        assert "response" in data
        assert "config" in data
        assert "judge" in data

        # Verify config keys
        assert "model" in data["config"]
        assert "layer" in data["config"]
        assert "strength" in data["config"]
        assert "magnitude" in data["config"]
        assert "vector_norm" in data["config"]
        assert "prompt_version" in data["config"]
        assert "trial" in data["config"]

        # Verify judge keys
        assert data["judge"] is not None
        assert "answer" in data["judge"]
        assert "coherent" in data["judge"]
        assert "detected_concept" in data["judge"]
        assert "reasoning" in data["judge"]

    def test_output_with_failed_judge(self) -> None:
        """Verify output format when judge fails."""
        config = SweepConfig(
            model="Qwen/Qwen2.5-3B-Instruct",
            layer=24,
            strength=2.0,
            magnitude=80.0,
            vector_norm=40.0,
            prompt_version="v2",
            trial=15,
        )
        record = TrialRecord(
            id="test_trial",
            concept="fear",
            was_injected=True,
            response="Some response",
            config=config,
            judge=None,
            judge_error="Rate limit exceeded after 3 retries",
        )

        data = record.model_dump()
        assert data["judge"] is None
        assert data["judge_error"] == "Rate limit exceeded after 3 retries"
