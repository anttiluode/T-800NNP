from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t800nnp import T800


def make_train(rng: np.random.Generator, kind: int, length: int = 44) -> np.ndarray:
    """Two sources with the same single-time distribution and opposite dynamics."""
    x = np.empty(length, dtype=float)
    x[0] = rng.integers(0, 2)
    persistence = 0.90 if kind == 0 else 0.10
    for t in range(1, length):
        stay = rng.random() < persistence
        x[t] = x[t - 1] if stay else 1.0 - x[t - 1]
    return x


def run(seed: int, dynamic: bool, train_episodes: int = 450, test_episodes: int = 200) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    model = T800(n_receivers=40, n_lanes=2, seed=seed, dynamic=dynamic)
    warmup = 7

    for _ in range(train_episodes):
        kind = int(rng.integers(0, 2))
        train = make_train(rng, kind)
        model.reset_episode_boundary()
        for t, x in enumerate(train):
            model.step(x, route_target=(kind if t >= warmup else None), learn=True)

    lane_ok = 0
    lane_n = 0
    pulse_ok = 0
    pulse_n = 0
    waveform_tp = waveform_fp = waveform_fn = 0
    for _ in range(test_episodes):
        kind = int(rng.integers(0, 2))
        train = make_train(rng, kind)
        model.reset_episode_boundary()
        for t, x in enumerate(train):
            out = model.step(x, route_target=None, learn=False)
            if t < warmup:
                continue
            lane_ok += int(out["lane"] == kind)
            lane_n += 1
            routed = np.asarray(out["routed_event"])
            truth = np.zeros(2); truth[kind] = x
            pred_b = routed > 0.5
            truth_b = truth > 0.5
            waveform_tp += int(np.sum(pred_b & truth_b))
            waveform_fp += int(np.sum(pred_b & ~truth_b))
            waveform_fn += int(np.sum(~pred_b & truth_b))
            if x > 0.5:
                pulse_n += 1
                pulse_ok += int(routed[kind] == 1.0)

    f1 = 2 * waveform_tp / max(1, 2 * waveform_tp + waveform_fp + waveform_fn)
    return {
        "route_accuracy": lane_ok / lane_n,
        "pulse_preservation": pulse_ok / max(1, pulse_n),
        "routed_waveform_f1": f1,
        "mean_used_capacity": float(np.mean(model.router.used_capacity)),
    }


def main() -> None:
    rows = {"dynamic": [], "static_current_only": []}
    for seed in range(10):
        rows["dynamic"].append(run(seed, True))
        rows["static_current_only"].append(run(seed, False))
    summary = {}
    for arm, vals in rows.items():
        summary[arm] = {}
        for key in vals[0]:
            a = np.array([r[key] for r in vals])
            summary[arm][key] = float(a.mean())
            summary[arm][key + "_std"] = float(a.std())
    print(json.dumps(summary, indent=2))
    out = ROOT / "results" / "gate1_temporal_router.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")

if __name__ == "__main__":
    main()
