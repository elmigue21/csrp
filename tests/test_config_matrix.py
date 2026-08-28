import ast
from pathlib import Path
import unittest


class ConfigMatrixTest(unittest.TestCase):
    def test_configs_cover_full_label_feature_lighting_factorial(self):
        source = Path("src/run_experiment.py").read_text()
        module = ast.parse(source)

        configs = None
        for node in module.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "CONFIGS":
                        configs = node.value
                        break

        self.assertIsNotNone(configs, "CONFIGS assignment not found")
        observed = set()
        for elt in configs.elts:
            self.assertIsInstance(elt, ast.Call)
            args = [ast.literal_eval(arg) for arg in elt.args[:4]]
            _, label_scheme, feature_scheme, include_lux = args
            observed.add((label_scheme, feature_scheme, include_lux))

        expected = {
            (label, feature, include_lux)
            for label in ("absolute", "within_person")
            for feature in ("raw", "within_person")
            for include_lux in (True, False)
        }
        self.assertEqual(expected, observed)


if __name__ == "__main__":
    unittest.main()
