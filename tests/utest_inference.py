import os
import unittest

from inference import run_detection

general_path = "data"

cat_data_path = os.path.join(general_path, "cat.jpg")
bicycle_data_path = os.path.join(general_path, "bicycle.jpg")
tabby_cat_data_path = os.path.join(general_path, "tabby_cat.jpeg")

cat_class = "cat"
bicycle_class = "bicycle"


class InferenceTest(unittest.TestCase):
    def test_classPredicted(self):
        class_name = run_detection(cat_data_path, show_res=False)[0]
        self.assertTrue(class_name and len(class_name) is 1)

    def test_catClassPredicted(self):
        self.assertEqual(cat_class, run_detection(cat_data_path, show_res=False)[0][0])

    def test_tabbyCatClassPredicted(self):
        self.assertEqual(cat_class, run_detection(tabby_cat_data_path, show_res=False)[0][0])

    def test_tabbyCatScorePassedThreshold(self):
        tabby_cat_threshold = 0.85
        self.assertTrue(run_detection(tabby_cat_data_path, show_res=False)[1][0] >= tabby_cat_threshold)

    def test_bicycleClassPredicted(self):
        self.assertEqual(bicycle_class, run_detection(bicycle_data_path, show_res=False)[0][0])

    def test_bicycleScorePassedThreshold(self):
        bicycle_threshold = 0.89
        self.assertTrue(run_detection(bicycle_data_path, show_res=False)[1][0] >= bicycle_threshold)


if __name__ == '__main__':
    unittest.main()
