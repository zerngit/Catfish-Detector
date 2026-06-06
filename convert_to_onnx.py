"""
Run this script ONCE locally to convert the LightGBM model to ONNX format.

    pip install onnxmltools skl2onnx
    python convert_to_onnx.py
"""

import joblib
import numpy as np
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

bundle = joblib.load("catfish_best_model.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns: list = bundle["feature_columns"]

n_features = len(feature_columns)

lgb_onnx = onnxmltools.convert_lightgbm(
    model,
    initial_types=[("float_input", FloatTensorType([None, n_features]))],
)
with open("model.onnx", "wb") as f:
    f.write(lgb_onnx.SerializeToString())

np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)
np.save("feature_columns.npy", np.array(feature_columns))

print(f"Done. Features: {n_features}")
print("Generated: model.onnx, scaler_mean.npy, scaler_scale.npy, feature_columns.npy")
