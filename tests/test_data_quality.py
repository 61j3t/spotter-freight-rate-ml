import unittest

import numpy as np
import pandas as pd

from src.data import clean_inputs, corrupted_label_mask, model_fit_index
from src.features import fit_schema
from src.features_native import FrequencyEncoder, build_native_static
from src.finalize import DECEMBER_OUTPUT_COLUMNS, _december_output_frame


class DataQualityTests(unittest.TestCase):
    def test_clean_inputs_repairs_weight_sign_and_keeps_indicators(self):
        frame = pd.DataFrame(
            {
                "weight": [-12_000.0, 8_000.0, np.nan, 0.0],
                "date": ["2025-01-01"] * 4,
            }
        )

        cleaned = clean_inputs(frame)

        self.assertEqual(cleaned["weight_was_negative"].tolist(), [1, 0, 0, 0])
        self.assertEqual(cleaned["weight_was_missing"].tolist(), [0, 0, 1, 0])
        self.assertEqual(cleaned.loc[0, "weight"], 12_000.0)
        self.assertTrue(pd.isna(cleaned.loc[2, "weight"]))
        self.assertTrue(pd.isna(cleaned.loc[3, "weight"]))

    def test_corruption_screen_separates_large_multiplicative_errors(self):
        distance = np.linspace(100.0, 2_000.0, 200)
        equipment = np.resize(np.array(["Dry Van", "Reefer", "Flatbed"]), 200)
        equipment_factor = pd.Series(equipment).map(
            {"Dry Van": 1.0, "Reefer": 1.08, "Flatbed": 1.15}
        ).to_numpy()
        rate = distance * 2.1 * equipment_factor
        rate[[7, 111]] *= np.array([3.0, 0.3])
        frame = pd.DataFrame(
            {
                "distance": distance,
                "equipment": equipment,
                "posted_rate": rate,
            }
        )

        flagged = corrupted_label_mask(frame)

        self.assertEqual(frame.index[flagged].tolist(), [7, 111])

    def test_model_fit_index_screens_only_candidate_training_rows(self):
        distance = np.linspace(100.0, 1_000.0, 30)
        frame = pd.DataFrame(
            {
                "distance": distance,
                "equipment": ["Dry Van"] * 30,
                "posted_rate": distance * 2.0,
            }
        )
        frame.loc[5, "posted_rate"] *= 4.0
        frame.loc[25, "posted_rate"] *= 4.0

        fit_index = model_fit_index(frame, frame.index[:20])

        self.assertNotIn(5, fit_index)
        self.assertNotIn(25, fit_index)
        self.assertTrue(set(fit_index).issubset(set(range(20))))

    def test_december_output_drops_internal_audit_columns(self):
        frame = pd.DataFrame(
            {
                "pickup": ["Lexington, KY"],
                "delivery": ["Fort Wayne, IN"],
                "distance": [360.0],
                "equipment": ["Dry Van"],
                "weight": [32_000.0],
                "date": [pd.Timestamp("2025-12-01")],
                "predicted_rate": [np.nan],
                "weight_was_negative": [0],
                "weight_was_missing": [0],
            }
        )

        output = _december_output_frame(frame, np.array([825.0]))

        self.assertEqual(output.columns.tolist(), DECEMBER_OUTPUT_COLUMNS)
        self.assertEqual(output.loc[0, "predicted_rate"], 825.0)

    def test_native_features_keep_categories_and_training_only_frequencies(self):
        frame = pd.DataFrame(
            {
                "pickup": ["A", "A", "B"],
                "delivery": ["X", "Y", "X"],
                "distance": [100.0, 200.0, 150.0],
                "equipment": ["Dry Van", "Reefer", "Dry Van"],
                "weight": [10_000.0, 20_000.0, 15_000.0],
                "date": pd.to_datetime(
                    ["2025-01-01", "2025-01-02", "2025-01-03"]
                ),
                "market_index": [1.0, 1.1, 0.9],
                "quote_signal": [2.0, 2.1, 1.9],
                "pickup_lat": [0.0, 0.0, 1.0],
                "pickup_lon": [0.0, 0.0, 1.0],
                "delivery_lat": [1.0, 2.0, 1.0],
                "delivery_lon": [1.0, 2.0, 1.0],
            }
        )

        schema = fit_schema(frame)
        native = build_native_static(frame, schema)
        self.assertEqual(str(native["lane_cat"].dtype), "category")

        encoder = FrequencyEncoder().fit(frame.iloc[:2])
        transformed = encoder.transform(frame.iloc[2:])
        self.assertEqual(float(transformed.iloc[0]["pickup_freq"]), 0.0)


if __name__ == "__main__":
    unittest.main()
