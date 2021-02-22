import unittest

class MyTestCase(unittest.TestCase):
    def test_something(self):
        from lingyi_pkg.categorical_variables import Simulator
        simulator = Simulator()
        simulator.generate_feature_levels()
        simulator.generate_noised_scores()
        simulator.generate_target()
        simulator.plot_score_density()


if __name__ == '__main__':
    unittest.main()
