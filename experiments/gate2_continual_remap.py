from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t800nnp import T800
from experiments.gate1_temporal_router import make_train


def evaluate(model: T800, rng: np.random.Generator, mapping: tuple[int,int], n_episodes: int = 100) -> float:
    ok = total = 0
    for _ in range(n_episodes):
        kind = int(rng.integers(0,2)); tr = make_train(rng, kind)
        model.reset_episode_boundary()
        for t,x in enumerate(tr):
            out = model.step(x, None, False)
            if t >= 7:
                ok += int(out["lane"] == mapping[kind]); total += 1
    return ok / total


def train_block(model: T800, rng: np.random.Generator, mapping: tuple[int,int], n_episodes: int) -> None:
    for _ in range(n_episodes):
        kind = int(rng.integers(0,2)); tr = make_train(rng, kind)
        model.reset_episode_boundary()
        for t,x in enumerate(tr):
            model.step(x, mapping[kind] if t >= 7 else None, True)


def run(seed: int) -> dict[str,float]:
    rng = np.random.default_rng(seed)
    model = T800(n_receivers=40, n_lanes=2, seed=seed, dynamic=True)
    map_a=(0,1); map_b=(1,0)
    train_block(model,rng,map_a,350)
    before = evaluate(model,rng,map_a,100)
    immediately_after_flip = evaluate(model,rng,map_b,60)

    checkpoints=[]
    for block in range(1,9):
        train_block(model,rng,map_b,35)
        checkpoints.append(evaluate(model,rng,map_b,50))
    after = checkpoints[-1]
    crossing = next((35*(i+1) for i,a in enumerate(checkpoints) if a>=0.80), None)
    return {
        "old_mapping_accuracy": before,
        "new_mapping_before_relearning": immediately_after_flip,
        "new_mapping_after_relearning": after,
        "episodes_to_80pct": float(crossing) if crossing is not None else float("nan"),
        "used_capacity": float(np.mean(model.router.used_capacity)),
    }


def main() -> None:
    rows=[run(seed) for seed in range(8)]
    summary={}
    for key in rows[0]:
        a=np.array([r[key] for r in rows],dtype=float)
        summary[key]=float(np.nanmean(a)); summary[key+"_std"]=float(np.nanstd(a))
    print(json.dumps(summary,indent=2))
    out=ROOT/"results"/"gate2_continual_remap.json"
    out.write_text(json.dumps(summary,indent=2)+"\n")

if __name__ == '__main__':
    main()
