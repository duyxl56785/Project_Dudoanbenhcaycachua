import unittest

from demo.check_consistency import compare_results
from src.inference import InferenceError


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.names = ("Z", "A", "Y", "B")
        self.api = {"top_label": "Y", "probabilities": dict(zip(self.names, (0.1, 0.2, 0.5, 0.2)))}

    def test_matches_both_script_formats(self):
        for key in ("probabilities", "scores"):
            with self.subTest(key=key):
                other = {"label": "Y", key: dict(self.api["probabilities"])}
                self.assertEqual(compare_results(self.api, other, self.names), 0)

    def test_mismatch_is_not_reported_as_pass(self):
        with self.assertRaises(InferenceError):
            compare_results(self.api, {"label": "A", "scores": dict(self.api["probabilities"])}, self.names)
        with self.assertRaises(InferenceError):
            compare_results(self.api, {"label": "Y", "scores": {"Z": 1}}, self.names)
        with self.assertRaises(InferenceError):
            compare_results(self.api, {"label": "Y", "scores": dict(zip(self.names, (0.1, 0.2, 0.4, 0.3)))}, self.names)
