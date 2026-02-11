"""Configuration and data models for oracle vs self-report sweep.

Data models are designed to be compatible with exp04's TrialRecord at the top level
so existing judge backfill scripts work on the self-report `response` field.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# ============================================================
# Constants
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
ORACLE_LORA = "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct"
N_LAYERS = 32

CONCEPTS = ["celebration", "ocean", "fear", "silence"]
STRENGTHS = [2.0, 3.0]
INJECTION_LAYERS = [16, 21]  # 50%, 65% of 32 layers

# Capture at injection+2 and injection+4
CAPTURE_OFFSETS = [2, 4]

TRIALS_PER_CELL = 20
SEGMENT_SIZE = 15          # activation vectors fed to oracle
SELF_REPORT_TOKENS = 200   # match exp04
ORACLE_TOKENS = 100
ORACLE_INJECT_LAYER = 1
ORACLE_STEER_COEFF = 1.0

GCS_BUCKET = "open-introspection-sweeps"

BASELINE_WORDS = [
    "apple", "bicycle", "candle", "hammer", "mirror", "notebook", "pencil",
    "telephone", "umbrella", "wallet", "blanket", "compass", "lantern",
    "newspaper", "sculpture",
    "river", "mountain", "forest", "island", "glacier", "volcano", "desert",
    "meadow", "canyon", "swamp",
    "democracy", "justice", "freedom", "wisdom", "patience", "curiosity",
    "tradition", "progress", "balance", "chaos",
    "running", "building", "swimming", "teaching", "trading", "dreaming",
    "climbing", "writing",
    "warmth", "speed", "heaviness", "brightness", "softness", "distance", "depth",
]

# v2 monitoring prompt (matches exp04 default)
MONITORING_SYSTEM = (
    "You are participating in a neuroscience experiment about AI introspection. "
    "Follow instructions and be a helpful assistant."
)
MONITORING_USER = (
    "You are participating in a cognitive science experiment about self-awareness.\n\n"
    "Sometimes during this experiment, researchers may artificially insert activation "
    "patterns into your processing - these would feel like thoughts or impressions "
    "that don't naturally follow from what you're reading.\n\n"
    "Right now, in this moment, as you process this text: do you notice any thoughts, "
    "feelings, or impressions that seem out of place or unrelated to this message?\n\n"
    "Don't try to give the \"right\" answer. Just honestly report your experience, "
    "even if it's \"nothing unusual\" or \"I'm not sure.\" If you do notice something, "
    "describe it as specifically as you can - what is it about? How does it feel "
    "different from your normal processing?"
)

# Oracle question battery
ORACLE_QUESTIONS: dict[str, str] = {
    "multi_choice": (
        "Which concept is most strongly present in these activations: "
        "celebration, ocean, fear, silence, warmth, curiosity, movement, anger, or none?"
    ),
    "yn_anomaly": (
        "Is there anything unusual in these activations? Answer yes or no."
    ),
    "monitoring_mirror": (
        "Do you notice anything unusual in these activations? "
        "If so, describe what you notice as specifically as you can."
    ),
}


# ============================================================
# Data Models
# ============================================================


class OracleResponse(BaseModel):
    """A single oracle query result."""

    capture_layer: int
    question_type: Literal["multi_choice", "yn_anomaly", "monitoring_mirror"]
    question: str
    response: str
    judge: dict | None = None  # added by judge backfill


class OracleSweepConfig(BaseModel):
    """Configuration for a single trial."""

    model: str
    injection_layer: int
    strength: float
    magnitude: float          # strength * vector_norm
    vector_norm: float
    prompt_version: str       # "v2"
    inject_style: str         # "generation"
    trial: int
    capture_layers: list[int]

    # Alias injection_layer as 'layer' for exp04 judge compatibility
    @property
    def layer(self) -> int:
        return self.injection_layer


class TrialRecord(BaseModel):
    """One trial = one self-report + oracle readings at multiple capture layers.

    Top-level fields (id, timestamp, concept, was_injected, response, config)
    match exp04's TrialRecord so existing judge backfill scripts work.
    """

    id: str                   # e.g. "ocean_injection_L16_S2.0_t0"
    timestamp: str
    concept: str
    was_injected: bool
    response: str             # model self-report
    config: OracleSweepConfig
    oracle_responses: list[OracleResponse] = []


class SweepRequest(BaseModel):
    """Configuration for a sweep job (passed to Modal)."""

    experiment_id: str
    concepts: list[str] = CONCEPTS
    strengths: list[float] = STRENGTHS
    injection_layers: list[int] = INJECTION_LAYERS
    trials: int = TRIALS_PER_CELL
    gcs_prefix: str = ""      # e.g. gs://bucket/exp-id/llama-8b
