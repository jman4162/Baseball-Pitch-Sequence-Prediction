"""Tests for the pitch simulator."""

import random

import pandas as pd
import pytest

from pitch_sequencing.data.simulator import (
    BaseballPitchSimulator,
    PITCHER_ARCHETYPES,
    generate_dataset,
    generate_hmm_sequences,
)


def test_generate_dataset_columns():
    """generate_dataset returns correct columns and dtypes."""
    df = generate_dataset(num_games=5, at_bats_per_game=5, seed=42)
    expected_cols = [
        "Balls", "Strikes", "PitchType", "Outcome",
        "PitcherType", "PitchNumber", "AtBatNumber",
        "RunnersOn", "ScoreDiff", "PreviousPitchType",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) > 0
    assert df["PitchType"].isin(["Fastball", "Slider", "Curveball", "Changeup"]).all()
    assert df["Outcome"].isin(["ball", "strike", "hit"]).all()


def test_pitch_distributions_differ_by_archetype():
    """Different pitcher archetypes produce different pitch distributions."""
    distributions = {}
    for ptype in PITCHER_ARCHETYPES:
        random.seed(42)
        sim = BaseballPitchSimulator(pitcher_type=ptype)
        pitches = []
        for _ in range(500):
            pitch_type, _ = sim.simulate_pitch((0, 0))
            pitches.append(pitch_type)
        dist = pd.Series(pitches).value_counts(normalize=True)
        distributions[ptype] = dist

    # Power pitcher should throw more fastballs than finesse
    assert distributions["power"].get("Fastball", 0) > distributions["finesse"].get("Fastball", 0)
    # Slider specialist should throw more sliders than balanced
    assert distributions["slider_specialist"].get("Slider", 0) > distributions["balanced"].get("Slider", 0)


def test_fatigue_shifts_probabilities():
    """Fatigue increases fastball rate by verifying the internal probability shift."""
    sim = BaseballPitchSimulator(pitcher_type="power")

    # Get fresh probabilities
    base_probs = sim.transition_probs[(1, 1)]["pitch_probs"]
    fresh_probs = sim._blend_pitch_probs(base_probs)
    fresh_probs = sim._apply_sequence_strategies(fresh_probs)
    fresh_outcomes = dict(sim.transition_probs[(1, 1)]["outcomes"]["Fastball"])
    fresh_fb = fresh_probs["Fastball"]

    # Apply fatigue
    sim.pitch_count = 120  # well past power's 85 threshold
    fatigued_probs, fatigued_outcomes = sim._apply_fatigue(dict(fresh_probs), fresh_outcomes)
    fatigued_fb = fatigued_probs["Fastball"]

    # Fatigued should have higher fastball probability
    assert fatigued_fb > fresh_fb
    # Fatigued should have higher ball probability
    assert fatigued_outcomes["ball"] > fresh_outcomes["ball"]


def test_generate_hmm_sequences_shape():
    """HMM sequence generator returns correct shape."""
    df = generate_hmm_sequences(num_sequences=10, sequence_length=20, seed=42)
    assert df.shape == (10, 20)
    assert all(c.startswith("Pitch_") for c in df.columns)
    assert df.isin(["Fastball", "Curveball", "Slider", "Changeup"]).all().all()
