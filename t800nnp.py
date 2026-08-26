from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np

Array = np.ndarray


def _project_l1(v: Array, budget: float) -> Array:
    """Project a vector onto the L1 ball ||v||_1 <= budget."""
    v = np.asarray(v, dtype=float)
    if np.sum(np.abs(v)) <= budget:
        return v
    u = np.abs(v)
    s = np.sort(u)[::-1]
    cssv = np.cumsum(s)
    idx = np.arange(1, len(v) + 1)
    cond = s - (cssv - budget) / idx > 0
    rho = idx[cond][-1]
    theta = (cssv[rho - 1] - budget) / rho
    return np.sign(v) * np.maximum(u - theta, 0.0)


@dataclass
class DynamicReceiverBank:
    """Persistent receiver-relative temporal coordinates.

    The same present input produces different coordinates when the receiver's
    history differs. Output events never reset these states.
    """

    n_receivers: int = 32
    seed: int = 0
    dynamic: bool = True

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.decay = rng.uniform(0.15, 0.985, self.n_receivers) if self.dynamic else np.zeros(self.n_receivers)
        self.state = np.zeros(self.n_receivers)
        self.gain = rng.uniform(0.8, 2.0, self.n_receivers)
        self.mix = rng.normal(size=(self.n_receivers, 4))
        self.bias = rng.normal(scale=0.15, size=self.n_receivers)

    @property
    def n_features(self) -> int:
        return self.n_receivers + 3

    def reset(self) -> None:
        self.state[:] = 0.0

    def step(self, x: float) -> Array:
        x = float(x)
        old = self.state.copy()
        self.state = self.decay * self.state + (1.0 - self.decay) * x
        transition = x - old if self.dynamic else np.zeros_like(old)
        local = np.column_stack((
            np.full(self.n_receivers, x),
            self.state,
            x * self.state,
            transition,
        ))
        phi = np.tanh(self.gain * np.sum(self.mix * local, axis=1) + self.bias)
        return np.concatenate(([x, float(np.mean(self.state)), float(np.mean(np.abs(transition)))], phi))


@dataclass
class LocalStructuralRouter:
    """Bounded local readout; no reverse graph traversal and no autograd.

    Each lane has a finite L1 structural budget. A lane's weight changes from
    its own local feature eligibility and its delayed scalar consequence only.
    """

    n_features: int
    n_lanes: int = 2
    seed: int = 0
    learning_rate: float = 0.0015
    eligibility_decay: float = 0.0
    consequence_delay: int = 0
    structural_budget: float = 20.0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.weight = rng.normal(scale=0.02, size=(self.n_lanes, self.n_features))
        self.eligibility = np.zeros_like(self.weight)
        self._error_queue: deque[Array] = deque()

    def reset_fast_state(self) -> None:
        self.eligibility[:] = 0.0
        self._error_queue.clear()

    def step(self, phi: Array, target: Array | None = None, learn: bool = True) -> Array:
        phi = np.asarray(phi, dtype=float)
        y = self.weight @ phi
        self.eligibility = self.eligibility_decay * self.eligibility + phi[None, :]
        if target is not None:
            err = np.asarray(target, dtype=float) - y
            self._error_queue.append(err.copy())
            if len(self._error_queue) > self.consequence_delay:
                delayed = self._error_queue.popleft()
                if learn:
                    for lane in range(self.n_lanes):
                        proposal = self.weight[lane] + self.learning_rate * delayed[lane] * self.eligibility[lane]
                        self.weight[lane] = _project_l1(proposal, self.structural_budget)
        return y

    @property
    def used_capacity(self) -> Array:
        return np.sum(np.abs(self.weight), axis=1)


@dataclass
class T800:
    """Continuously evolving signal-train router learned without backprop."""

    n_receivers: int = 32
    n_lanes: int = 2
    seed: int = 0
    dynamic: bool = True

    def __post_init__(self) -> None:
        self.receivers = DynamicReceiverBank(self.n_receivers, self.seed, self.dynamic)
        self.router = LocalStructuralRouter(self.receivers.n_features, self.n_lanes, self.seed + 1009)

    def reset_episode_boundary(self) -> None:
        # Independent experimental trials may reset. Emitting an event never does.
        self.receivers.reset()
        self.router.reset_fast_state()

    def step(self, x: float, route_target: int | None = None, learn: bool = True) -> dict[str, Array | int]:
        phi = self.receivers.step(x)
        target = None
        if route_target is not None:
            target = -np.ones(self.n_lanes)
            target[int(route_target)] = 1.0
        score = self.router.step(phi, target=target, learn=learn)
        lane = int(np.argmax(score))
        routed_event = np.zeros(self.n_lanes)
        routed_event[lane] = float(x)
        return {"phi": phi, "score": score, "lane": lane, "routed_event": routed_event}


@dataclass
class LeakySignalReceiver:
    """Minimal AIS-like receiver showing that continuation is receiver-relative."""

    decay: float = 0.85
    input_gain: float = 1.0
    threshold: float = 0.6
    state: float = 0.0

    def step(self, incoming_event: float) -> float:
        self.state = self.decay * self.state + self.input_gain * float(incoming_event)
        return float(self.state > self.threshold)
