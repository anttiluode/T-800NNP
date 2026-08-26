from experiments.gate0_signal_train import run as gate0
from experiments.gate1_temporal_router import main as gate1
from experiments.gate2_continual_remap import main as gate2
from experiments.gate3_temporal_xor import main as gate3

if __name__ == "__main__":
    print("\n=== GATE 0 ===")
    gate0()
    print("\n=== GATE 1 ===")
    gate1()
    print("\n=== GATE 2 ===")
    gate2()
    print("\n=== GATE 3 ===")
    gate3()
