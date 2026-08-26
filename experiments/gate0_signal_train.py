from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from t800nnp import LeakySignalReceiver


def run() -> dict:
    incoming = np.array([0,0,1,1,1,0,1,0,0,0,0,0,0,0], dtype=float)
    fast = LeakySignalReceiver(decay=0.25, input_gain=0.9, threshold=0.75)
    slow = LeakySignalReceiver(decay=0.82, input_gain=0.65, threshold=0.75)
    fast_out = np.array([fast.step(x) for x in incoming], dtype=int)
    slow_out = np.array([slow.step(x) for x in incoming], dtype=int)

    last_input = int(np.flatnonzero(incoming)[-1])
    last_fast = int(np.flatnonzero(fast_out)[-1]) if fast_out.any() else -1
    last_slow = int(np.flatnonzero(slow_out)[-1]) if slow_out.any() else -1
    result = {
        "incoming": incoming.astype(int).tolist(),
        "fast_receiver": fast_out.tolist(),
        "slow_receiver": slow_out.tolist(),
        "last_input_event": last_input,
        "fast_last_output_event": last_fast,
        "slow_last_output_event": last_slow,
        "fast_continuation_after_input": last_fast - last_input,
        "slow_continuation_after_input": last_slow - last_input,
    }
    print(json.dumps(result, indent=2))
    out = ROOT / "results" / "gate0_signal_train.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result

if __name__ == "__main__":
    run()
