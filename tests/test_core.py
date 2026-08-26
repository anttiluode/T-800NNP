import unittest
import numpy as np

from t800nnp import DynamicReceiverBank, LeakySignalReceiver, LocalStructuralRouter, T800


class CoreTests(unittest.TestCase):
    def test_dynamic_receiver_distinguishes_same_present_by_history(self):
        a = DynamicReceiverBank(n_receivers=16, seed=1, dynamic=True)
        b = DynamicReceiverBank(n_receivers=16, seed=1, dynamic=True)
        for x in [1, 1, 1, 1, 0]:
            pa = a.step(x)
        for x in [0, 0, 0, 0, 0]:
            pb = b.step(x)
        self.assertFalse(np.allclose(pa, pb))

    def test_structural_budget_is_bounded(self):
        r = LocalStructuralRouter(n_features=12, n_lanes=2, seed=2, structural_budget=2.0)
        for _ in range(100):
            r.step(np.ones(12), target=np.array([10.0, -10.0]), learn=True)
        self.assertTrue(np.all(r.used_capacity <= 2.0 + 1e-9))

    def test_output_event_does_not_reset_receiver_state(self):
        m = T800(n_receivers=8, seed=3)
        m.step(1.0, 0, True)
        m.step(0.0, 0, True)
        self.assertGreater(np.linalg.norm(m.receivers.state), 0.0)

    def test_receiver_can_continue_after_input_ends(self):
        r = LeakySignalReceiver(decay=0.85, input_gain=0.7, threshold=0.6)
        for _ in range(3):
            r.step(1.0)
        self.assertEqual(r.step(0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
