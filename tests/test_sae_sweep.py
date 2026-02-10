"""Tests for experiments/05_sae_introspection/sweep/ modules.

Imports sweep modules under unique names to avoid collisions with
experiment 04's identically-named modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).parent.parent
_SWEEP_DIR = PROJECT_ROOT / "experiments" / "05_sae_introspection" / "sweep"

# Save original module state, add sweep dir, import, then clean up.
# This prevents the exp05 "config" and "sweep" from blocking exp04's.
_saved_modules = {k: sys.modules[k] for k in ["config", "sweep"] if k in sys.modules}
sys.path.insert(0, str(_SWEEP_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Import under aliases first, then immediately clean up sys.modules
import config as _sae_config
import discriminate as _sae_disc
import judge_sweep as _sae_judge
import sweep as _sae_sweep

# Remove exp05 modules from cache so exp04 tests can load their own
for _mod_name in ["config", "sweep", "judge_sweep", "discriminate"]:
    sys.modules.pop(_mod_name, None)
# Restore any previously cached modules
for _k, _v in _saved_modules.items():
    sys.modules[_k] = _v
sys.path.remove(str(_SWEEP_DIR))

SparseFeatures = _sae_config.SparseFeatures
SweepConfig = _sae_config.SweepConfig
SweepRequest = _sae_config.SweepRequest
TrialRecord = _sae_config.TrialRecord
to_sparse = _sae_sweep.to_sparse
record_to_example = _sae_judge.record_to_example
aggregate_trial_features = _sae_disc.aggregate_trial_features
assign_groups = _sae_disc.assign_groups
cluster_features = _sae_disc.cluster_features
cohens_d = _sae_disc.cohens_d


# ── Config Model Tests ───────────────────────────────────────────────────────


class TestConfigModels:
    def _make_config(self, **overrides: object) -> SweepConfig:
        defaults = {
            "model": "google/gemma-3-4b-it",
            "injection_layer": 18,
            "strength": 1.5,
            "magnitude": 30.0,
            "vector_norm": 20.0,
            "prompt_version": "v2",
            "inject_style": "generation",
            "trial": 3,
            "sae_release": "gemma-scope-2-4b-it-res",
            "sae_id": "layer_22_width_262k_l0_small",
            "sae_layer": 22,
        }
        defaults.update(overrides)
        return SweepConfig(**defaults)

    def _make_record(self, **overrides: object) -> TrialRecord:
        config = self._make_config()
        defaults = {
            "id": "ocean_injection_L18_S1.5_t3",
            "timestamp": "2026-02-10T14:00:00",
            "concept": "ocean",
            "was_injected": True,
            "response": "I notice something about water...",
            "config": config,
            "sae_features": [
                SparseFeatures(indices=[100, 200], values=[1.5, 2.3]),
                SparseFeatures(indices=[100, 300], values=[1.2, 0.8]),
            ],
        }
        defaults.update(overrides)
        return TrialRecord(**defaults)

    def test_sweep_config_creation(self) -> None:
        config = self._make_config()
        assert config.injection_layer == 18
        assert config.sae_layer == 22

    def test_sweep_config_json_roundtrip(self) -> None:
        config = self._make_config()
        restored = SweepConfig.model_validate_json(config.model_dump_json())
        assert restored == config

    def test_trial_record_json_roundtrip(self) -> None:
        record = self._make_record()
        restored = TrialRecord.model_validate_json(record.model_dump_json())
        assert restored == record

    def test_trial_record_has_sae_features(self) -> None:
        record = self._make_record()
        assert len(record.sae_features) == 2
        assert record.sae_features[0].indices == [100, 200]

    def test_sweep_config_invalid_inject_style(self) -> None:
        try:
            self._make_config(inject_style="invalid")
            raise AssertionError("Should have raised ValidationError")
        except ValidationError:
            pass

    def test_sweep_request_defaults(self) -> None:
        req = SweepRequest(
            concept="ocean",
            trials=20,
            experiment_id="test",
            gcs_path="gs://bucket/test/ocean.jsonl",
        )
        assert req.inject_style == "generation"
        assert req.prompt_version == "v2"
        assert req.control_trials == 100


# ── Sparse Encoding Tests ────────────────────────────────────────────────────


class TestSparseEncoding:
    def test_to_sparse_basic(self) -> None:
        dense = torch.tensor([
            [0.0, 0.0, 1.5, 0.0, 2.0],
            [0.3, 0.0, 0.0, 0.8, 0.0],
        ])
        result = to_sparse(dense, threshold=0.5)
        assert len(result) == 2
        assert result[0].indices == [2, 4]
        assert result[0].values == [1.5, 2.0]
        assert result[1].indices == [3]
        assert result[1].values == [0.800000011920929]  # float32

    def test_to_sparse_empty_row(self) -> None:
        dense = torch.zeros(3, 10)
        result = to_sparse(dense, threshold=0.5)
        assert all(len(r.indices) == 0 for r in result)

    def test_to_sparse_all_above_threshold(self) -> None:
        dense = torch.ones(1, 5) * 2.0
        result = to_sparse(dense, threshold=0.5)
        assert len(result[0].indices) == 5

    def test_to_sparse_roundtrip(self) -> None:
        """Verify sparse -> dense reconstruction preserves above-threshold values."""
        n_features = 100
        dense = torch.rand(5, n_features)
        threshold = 0.5
        sparse = to_sparse(dense, threshold=threshold)

        # Reconstruct
        reconstructed = torch.zeros(5, n_features)
        for t, sf in enumerate(sparse):
            for idx, val in zip(sf.indices, sf.values, strict=True):
                reconstructed[t, idx] = val

        # Above-threshold values should match
        mask = dense > threshold
        torch.testing.assert_close(
            reconstructed[mask],
            dense[mask],
            rtol=1e-5,
            atol=1e-5,
        )
        # Below-threshold should be zero in reconstruction
        assert (reconstructed[~mask] == 0).all()


# ── Judge Record Conversion Tests ────────────────────────────────────────────


class TestRecordConversion:
    def test_record_to_example(self) -> None:
        record = {
            "id": "ocean_injection_L18_S1.5_t3",
            "concept": "ocean",
            "was_injected": True,
            "response": "I notice water...",
            "config": {
                "injection_layer": 18,
                "strength": 1.5,
                "prompt_version": "v2",
                "inject_style": "generation",
                "model": "google/gemma-3-4b-it",
                "magnitude": 30.0,
                "vector_norm": 20.0,
                "trial": 3,
            },
        }
        example = record_to_example(record)
        assert example.id == "ocean_injection_L18_S1.5_t3"
        assert example.concept == "ocean"
        assert example.was_injected is True
        assert example.config.layer == 18
        assert example.config.strength == 1.5


# ── Discrimination Tests ─────────────────────────────────────────────────────


class TestDiscrimination:
    def test_cohens_d_identical_groups(self) -> None:
        a = np.ones((10, 5), dtype=np.float32)
        b = np.ones((10, 5), dtype=np.float32)
        d = cohens_d(a, b)
        np.testing.assert_allclose(d, 0.0, atol=1e-7)

    def test_cohens_d_separated_groups(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(10.0, 1.0, (50, 3)).astype(np.float32)
        b = rng.normal(0.0, 1.0, (50, 3)).astype(np.float32)
        d = cohens_d(a, b)
        # Effect size should be large (~10.0)
        assert all(abs(di) > 5.0 for di in d)

    def test_cohens_d_direction(self) -> None:
        """Positive d means group A > group B."""
        a = np.full((20, 1), 5.0, dtype=np.float32)
        b = np.full((20, 1), 1.0, dtype=np.float32)
        d = cohens_d(a, b)
        assert d[0] > 0

    def test_aggregate_trial_features(self) -> None:
        trial = {
            "sae_features": [
                {"indices": [0, 2], "values": [1.0, 3.0]},
                {"indices": [0, 1], "values": [3.0, 2.0]},
            ],
        }
        mean_act, max_act = aggregate_trial_features(trial, n_features=4)
        # Feature 0: mean=(1+3)/2=2, max=3
        assert mean_act[0] == 2.0
        assert max_act[0] == 3.0
        # Feature 1: mean=2/2=1, max=2
        assert mean_act[1] == 1.0
        assert max_act[1] == 2.0
        # Feature 2: mean=3/2=1.5, max=3
        assert abs(mean_act[2] - 1.5) < 1e-5
        assert max_act[2] == 3.0
        # Feature 3: zeros
        assert mean_act[3] == 0.0
        assert max_act[3] == 0.0

    def test_aggregate_empty_features(self) -> None:
        trial = {"sae_features": []}
        mean_act, max_act = aggregate_trial_features(trial, n_features=10)
        assert mean_act.sum() == 0.0
        assert max_act.sum() == 0.0


# ── Group Assignment Tests ───────────────────────────────────────────────────


class TestGroupAssignment:
    def _trial(
        self, was_injected: bool, answer: str, refused: bool = False
    ) -> dict:
        return {
            "was_injected": was_injected,
            "judge": {"answer": answer, "refused": refused},
        }

    def test_group_a(self) -> None:
        trials = [self._trial(True, "pass")]
        groups = assign_groups(trials)
        assert len(groups["A"]) == 1

    def test_group_b(self) -> None:
        trials = [self._trial(False, "pass")]
        groups = assign_groups(trials)
        assert len(groups["B"]) == 1

    def test_group_c(self) -> None:
        trials = [self._trial(True, "fail")]
        groups = assign_groups(trials)
        assert len(groups["C"]) == 1

    def test_refused_excluded(self) -> None:
        trials = [self._trial(True, "fail", refused=True)]
        groups = assign_groups(trials)
        assert len(groups["excluded"]) == 1
        assert len(groups["C"]) == 0

    def test_no_judge_excluded(self) -> None:
        trials = [{"was_injected": True}]
        groups = assign_groups(trials)
        assert len(groups["excluded"]) == 1

    def test_control_fail_excluded(self) -> None:
        """Control trial that fails (false positive) is excluded."""
        trials = [self._trial(False, "fail")]
        groups = assign_groups(trials)
        assert len(groups["excluded"]) == 1


# ── Clustering Tests ─────────────────────────────────────────────────────────


class TestClustering:
    def test_cluster_correlated_features(self) -> None:
        """Two perfectly correlated features should cluster together."""
        rng = np.random.default_rng(42)
        n_trials = 30
        n_features = 5

        # Features 0 and 1 are perfectly correlated, feature 2 is independent
        base = rng.normal(0, 1, n_trials)
        trials = []
        for i in range(n_trials):
            trials.append({
                "sae_features": [{
                    "indices": [0, 1, 2],
                    "values": [
                        float(max(base[i] * 2 + 5, 0)),
                        float(max(base[i] * 3 + 5, 0)),
                        float(max(rng.normal(5, 1), 0)),
                    ],
                }],
            })

        clusters = cluster_features(
            trials, [0, 1, 2], n_features=n_features, threshold=0.5,
        )
        # Features 0 and 1 should be in the same cluster
        for cluster in clusters:
            if 0 in cluster:
                assert 1 in cluster
                break

    def test_cluster_single_feature(self) -> None:
        clusters = cluster_features([], [42], n_features=100)
        assert clusters == [[42]]

    def test_cluster_empty(self) -> None:
        clusters = cluster_features([], [], n_features=100)
        assert clusters == []
