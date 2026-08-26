# T-800NNP — trains carve routes

A falsification-first neural architecture experiment descended from:

- [GeoNeuronX](https://github.com/anttiluode/GeoNeuronX): history becomes receiver state; local nonlinearities act before collapse.
- [yrotisopeRweN](https://github.com/anttiluode/yrotisopeRweN): continuous traffic, eligibility, finite structure, recurrence, stable anonymous addresses.
- [Twensday](https://github.com/anttiluode/Twensday): what useful machine appears if we keep the dynamics instead of reducing everything to a static matrix?

T-800NNP asks:

> **Can continuously evolving receiver state learn to route temporal signal trains, preserve their events, adapt routes online, and perform temporal computations without backpropagation through the network or through time?**

This is not a biological neuron simulator and not a novelty claim. The code is deliberately small enough to attack.

## The primitive

```text
incoming temporal traffic
       ↓ broadcast
persistent receiver states q_j(t)
       ↓
stable local nonlinear coordinates phi_j(t)
       ↓
finite signed structural capacity per output lane
       ↓
receiver chooses continuation route
       ↓
original event is forwarded on that lane
```

The same present event can produce a different route because the receiver is already in a different state.

For receiver coordinate `j`:

```text
q_j(t+1) = a_j q_j(t) + (1-a_j) x(t)
phi_j(t) = nonlinear(x(t), q_j(t), x(t)q_j(t), x(t)-q_j(t-1))
```

The router is bounded:

```text
||w_lane||_1 <= C
```

so a lane cannot acquire unlimited structural strength.

## No backpropagation

There is no PyTorch, autograd, reverse graph traversal, gradient tape, or BPTT.

Gate 1/2/3 use an immediate local lane error. Gate 4 adds a stricter delayed-credit primitive:

```text
local receiver feature phi(t)
       ↓ retain its local address
six steps pass
       ↓
consequence for t arrives
       ↓
update only the old retained phi(t)
       ↓
project routing structure back into finite L1 capacity
```

`DelayedLocalStructuralRouter` literally retains old local feature vectors long enough for later consequence to address them. It does **not** differentiate backward through the intervening dynamics. This is still not a general solution to deep credit assignment: the environment supplies a useful route consequence.

The architectural distinction is:

> **the machine does not learn a matrix by differentiating a global loss through a stack; local receiver state creates the coordinates, and local consequence changes bounded routing structure. The matrix is something we can inspect afterward.**

## Gate 0 — signal trains really exist

The same finite incoming train is delivered to two receivers. One ends its outgoing train exactly with the input; the slower receiver continues firing for four steps after the final incoming event.

```text
incoming       0 0 1 1 1 0 1 0 0 0 0 0 0 0
fast receiver  0 0 1 1 1 0 1 0 0 0 0 0 0 0
slow receiver  0 0 0 1 1 1 1 1 1 1 1 0 0 0
```

So the sender determines when its own train stops, but the **receiver determines whether the perturbation continues as downstream traffic**.

## Gate 1 — route a train from its dynamics

Two hidden train families have the same one-time pulse probability (`P(x=1)=0.5`). They differ only in temporal law:

- family A strongly persists;
- family B strongly alternates.

The router sees only the pulse stream. After a short warmup it must choose a lane and forward the incoming pulse unchanged on the chosen lane.

10 seeds:

```text
dynamic receiver
route accuracy          0.8823 ± 0.0189
pulse preservation      0.8805 ± 0.0383
routed waveform F1      0.8805 ± 0.0383

current-only control
route accuracy          0.5144 ± 0.0357
pulse preservation      0.5183 ± 0.0320
```

Survives:

> **Present-time-identical events can be routed differently from receiver-carried history, and the original events can be forwarded rather than regenerated.**

It does not earn source separation in arbitrary mixtures.

## Gate 2 — online route remapping

Train temporal family A to lane 0 and B to lane 1. Then flip the rule without resetting weights.

8 seeds:

```text
old mapping accuracy                 0.8751 ± 0.0123
new mapping before relearning        0.1204 ± 0.0156
new mapping after relearning         0.8068 ± 0.0652
episodes to recover >= 0.80          128.3 ± 43.7
```

So the routes are not frozen lookup tables. Bounded local structure can be reallocated online.

## Gate 3 — temporal XOR

For iid binary input, predict:

```text
y(t) = x(t) XOR x(t-2)
```

8 seeds:

```text
dynamic receiver        0.7575 ± 0.0256
current-only control    0.5002 ± 0.0049
exact digital delay XOR 1.0000
```

The exact digital delay solution destroys the benchmark. What survives is smaller:

> **continuous receiver state plus local nonlinear coordinates can support a genuine nonlinear temporal operation, and a local bounded learner can read it out without backpropagation through time.**

## Gate 4 — NO EPISODES

This gate removes the experimental boundary that remained in Gate 1.

One binary stream runs for 40,000 steps. A hidden temporal law occasionally changes between:

```text
persistent   P(same next bit) = .90
alternating  P(same next bit) = .10
```

There is:

```text
NO reset at a hidden switch
NO start-of-train marker
NO end-of-train marker
NO change in one-time P(x=1)=.5
```

The receiver state runs continuously through all changes. The route consequence arrives **six steps late** and is credited to the retained local receiver address from six steps earlier. Learning is frozen after step 28,000; evaluation uses the remaining uninterrupted stream.

10 seeds:

```text
dynamic, no reset
route accuracy                 0.8829 ± 0.0119
settled route accuracy         0.8848 ± 0.0134
pulse route accuracy           0.8917 ± 0.0131

current-only, no reset
route accuracy                 0.4824 ± 0.0322

shuffled delayed consequence
route accuracy                 0.4825 ± 0.0905

explicit smoothed transition attacker
route accuracy                 0.9723 ± 0.0026
```

So the no-reset system still does useful temporal routing, and breaking delayed causal address destroys learning. The explicit temporal statistic remains much better.

Survives:

> **A continuously running receiver can infer changing temporal traffic without episode boundaries, and delayed local consequence can carve useful routing structure without BPTT.**

Does **not** survive:

> a claim that the architecture is the best way to solve this temporal inference problem.

The explicit Markov-style attacker wins badly.

## Run

```bash
python -m pip install -r requirements.txt
python experiments/gate0_signal_train.py
python experiments/gate1_temporal_router.py
python experiments/gate2_continual_remap.py
python experiments/gate3_temporal_xor.py
python experiments/gate4_no_episode_stream.py
python run_all.py
python -m unittest discover -s tests
```

Results are written to `results/`.

## What has NOT been earned

- biological fidelity;
- deep credit assignment without a teaching/consequence signal;
- superiority to RNNs, SSMs, reservoirs, spiking networks, or explicit temporal statistics;
- blind source separation of arbitrary simultaneous mixtures;
- learned multi-hop routing through a large recurrent population;
- spontaneous discovery of start/end boundaries as discrete objects;
- proof that the structural budget is better than ordinary signed weights.

## Next clean attacks

1. **Multi-hop population.** Broadcast traffic into many continuously running receivers with many possible paths. Reward only the destination and see whether a selective subgraph grows.
2. **True mixtures.** Superpose independent temporal train families and test whether local history can separate/reroute events without a supplied source identity at every step.
3. **Boundary emergence.** Add quiet gaps, overlapping bursts and ambiguous switches; ask whether downstream continuation itself supplies an operational notion of train start/end.
4. **Cut test.** After learning, remove high-capacity paths and verify the operation dies specifically there.
5. **Ordinary attackers.** Reservoir + linear readout, small GRU/RNN, explicit Markov statistic, and exact delay-feature models.
6. **Matrix autopsy.** Inspect rank, sparsity, singular spectrum, and state-conditioned effective operators only after the dynamical machine has learned them.

The next milestone is no longer merely "remove the reset". Gate 4 did that.

It is:

> **continuous temporal traffic carving a selective multi-hop route by delayed local credit while the network never stops running.**
