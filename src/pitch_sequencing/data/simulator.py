"""Baseball pitch simulator with pitcher archetypes, sequence strategies, and fatigue."""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PITCHER_ARCHETYPES = {
    "power": {
        "pitch_bias": {"Fastball": 0.55, "Slider": 0.25, "Curveball": 0.10, "Changeup": 0.10},
        "fatigue_resistance": 85,
        "strike_tendency": 0.05,
    },
    "finesse": {
        "pitch_bias": {"Fastball": 0.25, "Slider": 0.20, "Curveball": 0.30, "Changeup": 0.25},
        "fatigue_resistance": 95,
        "strike_tendency": -0.03,
    },
    "slider_specialist": {
        "pitch_bias": {"Fastball": 0.35, "Slider": 0.40, "Curveball": 0.10, "Changeup": 0.15},
        "fatigue_resistance": 80,
        "strike_tendency": 0.02,
    },
    "balanced": {
        "pitch_bias": {"Fastball": 0.30, "Slider": 0.25, "Curveball": 0.25, "Changeup": 0.20},
        "fatigue_resistance": 90,
        "strike_tendency": 0.0,
    },
}

PITCH_SEQUENCE_STRATEGIES = [
    (["Fastball", "Fastball"], "Changeup", 0.20),
    (["Fastball", "Fastball"], "Slider", 0.15),
    (["Fastball", "Slider"], "Curveball", 0.20),
    (["Curveball"], "Fastball", 0.15),
    (["Changeup"], "Fastball", 0.18),
    (["Slider", "Slider"], "Fastball", 0.25),
    (["Curveball", "Curveball"], "Fastball", 0.20),
    (["Fastball", "Curveball"], "Slider", 0.15),
]

TRANSITION_PROBS: Dict[Tuple[int, int], dict] = {
    (0, 0): {
        "pitch_probs": {"Fastball": 0.40, "Slider": 0.25, "Curveball": 0.20, "Changeup": 0.15},
        "outcomes": {
            "Fastball": {"ball": 0.30, "strike": 0.58, "hit": 0.12},
            "Slider": {"ball": 0.40, "strike": 0.50, "hit": 0.10},
            "Curveball": {"ball": 0.48, "strike": 0.44, "hit": 0.08},
            "Changeup": {"ball": 0.38, "strike": 0.52, "hit": 0.10},
        },
    },
    (0, 1): {
        "pitch_probs": {"Fastball": 0.35, "Slider": 0.28, "Curveball": 0.20, "Changeup": 0.17},
        "outcomes": {
            "Fastball": {"ball": 0.28, "strike": 0.62, "hit": 0.10},
            "Slider": {"ball": 0.35, "strike": 0.55, "hit": 0.10},
            "Curveball": {"ball": 0.42, "strike": 0.50, "hit": 0.08},
            "Changeup": {"ball": 0.33, "strike": 0.58, "hit": 0.09},
        },
    },
    (0, 2): {
        "pitch_probs": {"Fastball": 0.30, "Slider": 0.32, "Curveball": 0.20, "Changeup": 0.18},
        "outcomes": {
            "Fastball": {"ball": 0.32, "strike": 0.62, "hit": 0.06},
            "Slider": {"ball": 0.38, "strike": 0.56, "hit": 0.06},
            "Curveball": {"ball": 0.45, "strike": 0.50, "hit": 0.05},
            "Changeup": {"ball": 0.40, "strike": 0.54, "hit": 0.06},
        },
    },
    (1, 0): {
        "pitch_probs": {"Fastball": 0.42, "Slider": 0.23, "Curveball": 0.20, "Changeup": 0.15},
        "outcomes": {
            "Fastball": {"ball": 0.28, "strike": 0.58, "hit": 0.14},
            "Slider": {"ball": 0.38, "strike": 0.50, "hit": 0.12},
            "Curveball": {"ball": 0.45, "strike": 0.45, "hit": 0.10},
            "Changeup": {"ball": 0.35, "strike": 0.53, "hit": 0.12},
        },
    },
    (1, 1): {
        "pitch_probs": {"Fastball": 0.38, "Slider": 0.25, "Curveball": 0.20, "Changeup": 0.17},
        "outcomes": {
            "Fastball": {"ball": 0.30, "strike": 0.58, "hit": 0.12},
            "Slider": {"ball": 0.38, "strike": 0.50, "hit": 0.12},
            "Curveball": {"ball": 0.44, "strike": 0.46, "hit": 0.10},
            "Changeup": {"ball": 0.36, "strike": 0.52, "hit": 0.12},
        },
    },
    (1, 2): {
        "pitch_probs": {"Fastball": 0.32, "Slider": 0.30, "Curveball": 0.20, "Changeup": 0.18},
        "outcomes": {
            "Fastball": {"ball": 0.30, "strike": 0.63, "hit": 0.07},
            "Slider": {"ball": 0.36, "strike": 0.57, "hit": 0.07},
            "Curveball": {"ball": 0.43, "strike": 0.51, "hit": 0.06},
            "Changeup": {"ball": 0.38, "strike": 0.55, "hit": 0.07},
        },
    },
    (2, 0): {
        "pitch_probs": {"Fastball": 0.45, "Slider": 0.22, "Curveball": 0.18, "Changeup": 0.15},
        "outcomes": {
            "Fastball": {"ball": 0.25, "strike": 0.57, "hit": 0.18},
            "Slider": {"ball": 0.35, "strike": 0.50, "hit": 0.15},
            "Curveball": {"ball": 0.42, "strike": 0.44, "hit": 0.14},
            "Changeup": {"ball": 0.33, "strike": 0.52, "hit": 0.15},
        },
    },
    (2, 1): {
        "pitch_probs": {"Fastball": 0.40, "Slider": 0.25, "Curveball": 0.18, "Changeup": 0.17},
        "outcomes": {
            "Fastball": {"ball": 0.28, "strike": 0.57, "hit": 0.15},
            "Slider": {"ball": 0.36, "strike": 0.50, "hit": 0.14},
            "Curveball": {"ball": 0.43, "strike": 0.45, "hit": 0.12},
            "Changeup": {"ball": 0.35, "strike": 0.51, "hit": 0.14},
        },
    },
    (2, 2): {
        "pitch_probs": {"Fastball": 0.35, "Slider": 0.28, "Curveball": 0.20, "Changeup": 0.17},
        "outcomes": {
            "Fastball": {"ball": 0.30, "strike": 0.62, "hit": 0.08},
            "Slider": {"ball": 0.37, "strike": 0.55, "hit": 0.08},
            "Curveball": {"ball": 0.44, "strike": 0.49, "hit": 0.07},
            "Changeup": {"ball": 0.38, "strike": 0.54, "hit": 0.08},
        },
    },
    (3, 0): {
        "pitch_probs": {"Fastball": 0.50, "Slider": 0.20, "Curveball": 0.15, "Changeup": 0.15},
        "outcomes": {
            "Fastball": {"ball": 0.22, "strike": 0.55, "hit": 0.23},
            "Slider": {"ball": 0.30, "strike": 0.50, "hit": 0.20},
            "Curveball": {"ball": 0.38, "strike": 0.44, "hit": 0.18},
            "Changeup": {"ball": 0.28, "strike": 0.52, "hit": 0.20},
        },
    },
    (3, 1): {
        "pitch_probs": {"Fastball": 0.48, "Slider": 0.22, "Curveball": 0.15, "Changeup": 0.15},
        "outcomes": {
            "Fastball": {"ball": 0.25, "strike": 0.57, "hit": 0.18},
            "Slider": {"ball": 0.33, "strike": 0.50, "hit": 0.17},
            "Curveball": {"ball": 0.40, "strike": 0.45, "hit": 0.15},
            "Changeup": {"ball": 0.30, "strike": 0.53, "hit": 0.17},
        },
    },
    (3, 2): {
        "pitch_probs": {"Fastball": 0.42, "Slider": 0.25, "Curveball": 0.18, "Changeup": 0.15},
        "outcomes": {
            "Fastball": {"ball": 0.28, "strike": 0.60, "hit": 0.12},
            "Slider": {"ball": 0.35, "strike": 0.53, "hit": 0.12},
            "Curveball": {"ball": 0.42, "strike": 0.47, "hit": 0.11},
            "Changeup": {"ball": 0.33, "strike": 0.55, "hit": 0.12},
        },
    },
}


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class BaseballPitchSimulator:
    def __init__(self, transition_probs: Optional[Dict] = None, pitcher_type: str = "balanced"):
        self.states = [(b, s) for b in range(4) for s in range(3)]
        self.pitch_types = ["Fastball", "Slider", "Curveball", "Changeup"]
        self.transition_probs = transition_probs or TRANSITION_PROBS
        self.pitcher_type = pitcher_type
        self.archetype = PITCHER_ARCHETYPES[pitcher_type]
        self.pitch_count = 0
        self.recent_pitches: List[str] = []

    def _blend_pitch_probs(self, base_probs: Dict[str, float]) -> Dict[str, float]:
        bias = self.archetype["pitch_bias"]
        blended = {p: 0.4 * base_probs[p] + 0.6 * bias[p] for p in self.pitch_types}
        total = sum(blended.values())
        return {p: v / total for p, v in blended.items()}

    def _apply_sequence_strategies(self, probs: Dict[str, float]) -> Dict[str, float]:
        if not self.recent_pitches:
            return probs
        probs = dict(probs)
        for prefix, next_pitch, boost in PITCH_SEQUENCE_STRATEGIES:
            n = len(prefix)
            if len(self.recent_pitches) >= n and self.recent_pitches[-n:] == prefix:
                probs[next_pitch] += boost
        total = sum(probs.values())
        return {p: v / total for p, v in probs.items()}

    def _apply_fatigue(
        self, probs: Dict[str, float], outcome_probs: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        threshold = self.archetype["fatigue_resistance"]
        if self.pitch_count <= threshold:
            return probs, outcome_probs

        fatigue = min((self.pitch_count - threshold) / 40.0, 0.4)

        probs = dict(probs)
        probs["Fastball"] += fatigue * 0.5
        for p in ["Slider", "Curveball", "Changeup"]:
            probs[p] = max(probs[p] - fatigue * 0.5 / 3, 0.02)
        total = sum(probs.values())
        probs = {p: v / total for p, v in probs.items()}

        outcome_probs = dict(outcome_probs)
        ball_boost = fatigue * 0.15
        outcome_probs["ball"] = min(outcome_probs["ball"] + ball_boost, 0.60)
        outcome_probs["strike"] = max(outcome_probs["strike"] - ball_boost * 0.7, 0.20)
        outcome_probs["hit"] = max(outcome_probs["hit"] - ball_boost * 0.3, 0.03)
        total = sum(outcome_probs.values())
        outcome_probs = {k: v / total for k, v in outcome_probs.items()}
        return probs, outcome_probs

    def _apply_situation(
        self,
        probs: Dict[str, float],
        outcome_probs: Dict[str, float],
        runners_on: bool,
        score_diff: int,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        probs = dict(probs)
        outcome_probs = dict(outcome_probs)

        if runners_on:
            probs["Fastball"] += 0.08
            probs["Curveball"] = max(probs["Curveball"] - 0.04, 0.02)
            probs["Changeup"] = max(probs["Changeup"] - 0.04, 0.02)

        if score_diff >= 3:
            outcome_probs["strike"] += 0.05
            outcome_probs["ball"] = max(outcome_probs["ball"] - 0.05, 0.10)
        elif score_diff <= -3:
            outcome_probs["ball"] += 0.05
            outcome_probs["strike"] = max(outcome_probs["strike"] - 0.05, 0.20)

        total_p = sum(probs.values())
        probs = {p: v / total_p for p, v in probs.items()}
        total_o = sum(outcome_probs.values())
        outcome_probs = {k: v / total_o for k, v in outcome_probs.items()}
        return probs, outcome_probs

    def simulate_pitch(
        self,
        current_state: Tuple[int, int],
        runners_on: bool = False,
        score_diff: int = 0,
    ) -> Tuple[str, str]:
        base_pitch_probs = self.transition_probs[current_state]["pitch_probs"]
        pitch_probs = self._blend_pitch_probs(base_pitch_probs)
        pitch_probs = self._apply_sequence_strategies(pitch_probs)

        pitch_types = list(pitch_probs.keys())
        pitch_type = random.choices(pitch_types, weights=list(pitch_probs.values()), k=1)[0]

        outcome_probs = dict(self.transition_probs[current_state]["outcomes"][pitch_type])
        outcome_probs["strike"] += self.archetype["strike_tendency"]
        outcome_probs["ball"] -= self.archetype["strike_tendency"]

        pitch_probs, outcome_probs = self._apply_fatigue(pitch_probs, outcome_probs)
        pitch_probs, outcome_probs = self._apply_situation(pitch_probs, outcome_probs, runners_on, score_diff)

        total = sum(outcome_probs.values())
        outcome_probs = {k: max(v / total, 0.01) for k, v in outcome_probs.items()}
        total = sum(outcome_probs.values())
        outcome_probs = {k: v / total for k, v in outcome_probs.items()}

        outcome = random.choices(list(outcome_probs.keys()), weights=list(outcome_probs.values()), k=1)[0]
        self.pitch_count += 1
        self.recent_pitches.append(pitch_type)
        return pitch_type, outcome

    def update_count(self, current_state, outcome):
        balls, strikes = current_state
        if outcome == "ball":
            balls += 1
            if balls == 4:
                return "walk"
        elif outcome == "strike":
            strikes += 1
            if strikes == 3:
                return "strikeout"
        elif outcome == "hit":
            return "hit"
        return (balls, strikes) if (balls < 4 and strikes < 3) else None

    def simulate_at_bat(self, runners_on: bool = False, score_diff: int = 0):
        state = (0, 0)
        sequence = []
        while True:
            pitch_type, outcome = self.simulate_pitch(state, runners_on, score_diff)
            sequence.append((state, pitch_type, outcome))
            new_state = self.update_count(state, outcome)
            if isinstance(new_state, tuple):
                state = new_state
            else:
                sequence.append(new_state)
                break
        return sequence


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(num_games: int = 3000, at_bats_per_game: int = 35, seed: int = 42) -> pd.DataFrame:
    """Generate the main pitch dataset by simulating full games."""
    random.seed(seed)
    pitcher_types = list(PITCHER_ARCHETYPES.keys())
    data = []

    for _ in range(num_games):
        pitcher_type = random.choice(pitcher_types)
        simulator = BaseballPitchSimulator(pitcher_type=pitcher_type)
        score_diff = 0

        for at_bat_num in range(1, at_bats_per_game + 1):
            runners_on = random.random() < 0.35
            if random.random() < 0.15:
                score_diff += random.choice([-1, 1, 1, 2])
            score_diff = max(min(score_diff, 8), -8)

            at_bat = simulator.simulate_at_bat(runners_on=runners_on, score_diff=score_diff)
            for item in at_bat[:-1]:
                state, pitch_type, outcome = item
                balls, strikes = state
                data.append([
                    balls, strikes, pitch_type, outcome,
                    pitcher_type, simulator.pitch_count,
                    at_bat_num, int(runners_on), score_diff,
                ])

    df = pd.DataFrame(data, columns=[
        "Balls", "Strikes", "PitchType", "Outcome",
        "PitcherType", "PitchNumber", "AtBatNumber",
        "RunnersOn", "ScoreDiff",
    ])
    df["PreviousPitchType"] = df["PitchType"].shift(1).fillna("None")
    return df


def generate_hmm_sequences(
    num_sequences: int = 2500,
    sequence_length: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the HMM synthetic pitch sequences dataset."""
    pitch_types = {0: "Fastball", 1: "Curveball", 2: "Slider", 3: "Changeup"}
    num_pitches = len(pitch_types)

    transition_matrix = np.array([
        [0.15, 0.20, 0.30, 0.35],
        [0.45, 0.10, 0.25, 0.20],
        [0.35, 0.30, 0.10, 0.25],
        [0.50, 0.15, 0.25, 0.10],
    ])

    np.random.seed(seed)
    sequences = []
    for _ in range(num_sequences):
        start = np.random.randint(num_pitches)
        seq = [start]
        current = start
        for _ in range(sequence_length - 1):
            nxt = np.random.choice(num_pitches, p=transition_matrix[current])
            seq.append(nxt)
            current = nxt
        sequences.append(seq)

    df = pd.DataFrame(sequences, columns=[f"Pitch_{i+1}" for i in range(sequence_length)])
    df.replace(pitch_types, inplace=True)
    return df
