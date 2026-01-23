"""Pydantic schemas for introspection judge."""

from typing import Literal

from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    """Configuration from the original experiment run."""

    layer: int
    strength: float
    prompt_version: str


class Label(BaseModel):
    """Human (or judge) label for an example."""

    answer: Literal["pass", "fail"] | None = None
    coherent: bool | None = None
    detected_concept: str | None = None
    labeler: str | None = None
    timestamp: str | None = None
    use_as_fewshot: bool | None = None
    fewshot_note: str | None = None


class Example(BaseModel):
    """A single labeled example for the introspection judge."""

    id: str
    source_file: str
    concept: str
    was_injected: bool
    response: str
    config: ExperimentConfig
    label: Label


class JudgeResult(BaseModel):
    """Output from the introspection judge."""

    reasoning: str
    answer: Literal["pass", "fail"]
    coherent: bool
    detected_concept: str | None = None
