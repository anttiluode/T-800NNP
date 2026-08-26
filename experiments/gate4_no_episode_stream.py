from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from t800nnp import ContinuousT800


def make_unbroken_stream(
    seed: int,
    n_steps: int = 40_000,
    switch_hazard: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """One uninterrupted binary stream with hidden changes in temporal law.

    Family 0 persists (90% chance to keep the previous bit).
    Family 1 alternates (90% chance to change the previous bit).

    Both have the same stationary one-time distribution P(x=1)=0.5. The hidden
    family changes with no boundary marker and the receiver is never reset.
    """
    rng = np.random.default_rng(seed)
    family = np.empty(n_steps, dtype=int)
    pulse = np.empty(n_steps, dtype=float)
    family[0] = int(rng.integers(0, 2))
    pulse[0] = float(rng.integers(0, 2))

    for t in range(1, n_steps):
        if rng.random() < switch_hazard:
            family[t] = 1 - family[t - 1]
        else:
            family[t] = family[t - 1]

        p_stay = 0.90 if family[t] == 0 else 0.10
        pulse[t] = pulse[t - 1] if rng.random() < p_stay else 1.0 - pulse[t - 1]

    return pulse, family


def smoothed_transition_oracle(
    pulse: np.ndarray,
    family: np.ndarray,
    eval_start: int,
    alpha: float = 0.80,
) -> float:
    """Ordinary explicit attacker: smooth whether the stream alternated."""
    score = 0.5
    pred = np.zeros(len(pulse), dtype=int)
    for t in range(1, len(pulse)):
        alternated = float(pulse[t] != pulse[t - 1])
        score = alpha * score + (1.0 - alpha) * alternated
        pred[t] = int(score > 0.5)
    return float(np.mean(pred[eval_start:] == family[eval_start:]))


def run(
    seed: int,
    *,
    dynamic: bool,
    shuffle_consequence: bool = False,
    n_steps: int = 40_000,
    train_steps: int = 28_000,
    consequence_delay: int = 6,
    switch_hazard: float = 0.01,
) -> dict[str, float]:
    pulse, family = make_unbroken_stream(seed, n_steps, switch_hazard)
    model = ContinuousT800(
        n_receivers=40,
        n_lanes=2,
        seed=seed,
        dynamic=dynamic,
        consequence_delay=consequence_delay,
        learning_rate=0.001,
    )
    rng = np.random.default_rng(seed + 9911)
    lane = np.empty(n_steps, dtype=int)

    for t in range(n_steps):
        delayed_target = None
        if consequence_delay <= t < train_steps + consequence_delay:
            delayed_target = int(family[t - consequence_delay])
            if shuffle_consequence:
                delayed_target = int(rng.integers(0, 2))

        out = model.step(
            pulse[t],
            delayed_route_target=delayed_target,
            learn=t < train_steps + consequence_delay,
        )
        lane[t] = int(out["lane"])

    # Nothing here resets at a hidden family switch. 'Settled' merely excludes
    # the first few post-switch steps from one diagnostic; the model is not told
    # where those switches happened.
    since_switch = np.zeros(n_steps, dtype=int)
    last_switch = 0
    for t in range(1, n_steps):
        if family[t] != family[t - 1]:
            last_switch = t
        since_switch[t] = t - last_switch

    idx = np.arange(n_steps)
    eval_mask = idx >= train_steps + consequence_delay
    settled_mask = eval_mask & (since_switch >= consequence_delay + 2)
    pulse_mask = eval_mask & (pulse > 0.5)

    return {
        "route_accuracy": float(np.mean(lane[eval_mask] == family[eval_mask])),
        "settled_route_accuracy": float(np.mean(lane[settled_mask] == family[settled_mask])),
        "pulse_route_accuracy": float(np.mean(lane[pulse_mask] == family[pulse_mask])),
        "mean_used_capacity": float(np.mean(model.router.used_capacity)),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in rows[0]:
        values = np.array([row[key] for row in rows], dtype=float)
        out[key] = float(values.mean())
        out[key + "_std"] = float(values.std())
    return out


def main() -> None:
    seeds = range(10)
    settings = {
        "n_seeds": 10,
        "n_steps": 40_000,
        "train_steps": 28_000,
        "consequence_delay": 6,
        "switch_hazard": 0.01,
    }

    dynamic_rows = [run(seed, dynamic=True) for seed in seeds]
    static_rows = [run(seed, dynamic=False) for seed in seeds]
    shuffled_rows = [run(seed, dynamic=True, shuffle_consequence=True) for seed in seeds]

    oracle = []
    for seed in seeds:
        pulse, family = make_unbroken_stream(seed, settings["n_steps"], settings["switch_hazard"])
        oracle.append(
            smoothed_transition_oracle(
                pulse,
                family,
                settings["train_steps"] + settings["consequence_delay"],
            )
        )

    summary = {
        "dynamic_no_reset": summarize(dynamic_rows),
        "current_only_no_reset": summarize(static_rows),
        "shuffled_delayed_consequence": summarize(shuffled_rows),
        "smoothed_transition_oracle": {
            "route_accuracy": float(np.mean(oracle)),
            "route_accuracy_std": float(np.std(oracle)),
        },
        "settings": settings,
    }

    print("\n=== GATE 4: NO EPISODES / CONTINUOUS TRAFFIC ===")
    print("Hidden temporal law changes without a boundary marker; receiver state never resets.\n")
    print(json.dumps(summary, indent=2))

    out = ROOT / "results" / "gate4_no_episode_stream.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
