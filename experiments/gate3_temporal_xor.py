from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from t800nnp import T800


def run(seed:int,dynamic:bool,n_train:int=30000,n_test:int=12000,delay:int=2)->float:
    rng=np.random.default_rng(seed)
    model=T800(n_receivers=64,n_lanes=2,seed=seed,dynamic=dynamic)
    history=[]
    for t in range(n_train):
        x=float(rng.integers(0,2)); history.append(int(x))
        target=None if t<delay else (history[t] ^ history[t-delay])
        model.step(x,target,True)
    ok=total=0
    for t in range(n_test):
        x=float(rng.integers(0,2)); history.append(int(x))
        idx=n_train+t
        target=history[idx] ^ history[idx-delay]
        out=model.step(x,None,False)
        ok += int(out['lane']==target); total+=1
    return ok/total


def main():
    rows={"dynamic":[],"static_current_only":[]}
    for seed in range(8):
        rows['dynamic'].append(run(seed,True)); rows['static_current_only'].append(run(seed,False))
    summary={k:{'accuracy':float(np.mean(v)),'accuracy_std':float(np.std(v))} for k,v in rows.items()}
    summary['exact_digital_delay_oracle']={'accuracy':1.0,'accuracy_std':0.0}
    print(json.dumps(summary,indent=2))
    (ROOT/'results'/'gate3_temporal_xor.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':
    main()
