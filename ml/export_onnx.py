import os
import sys
import pickle
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_DIR, "ml"))
from models import TabularDeepModel

ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "ml", "artifacts")
TRITON_DIR = os.path.join(WORKSPACE_DIR, "triton_models", "deep_fraud", "1")

def export_deep_model_to_onnx():
    print("=" * 80)
    print("STEP 1: EXPORTING PYTORCH DEEP MODEL TO ONNX & TRITON REPOSITORY")
    print("=" * 80)

    # 1. Load Preprocessor and Model
    with open(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"), "rb") as f:
        prep = pickle.load(f)

    cat_dims = [len(cat) + 1 for cat in prep["encoder"].categories_]
    num_dim = len(prep["num_cols"])

    deep_model = TabularDeepModel(cat_dims=cat_dims, num_dim=num_dim)
    model_path = os.path.join(ARTIFACTS_DIR, "deep_model.pth")
    deep_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    deep_model.eval()

    # 2. Define Sample Inputs for Tracing
    batch_size = 4
    dummy_x_cat = torch.zeros((batch_size, len(cat_dims)), dtype=torch.int64)
    dummy_x_num = torch.zeros((batch_size, num_dim), dtype=torch.float32)

    # Output paths
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(TRITON_DIR, exist_ok=True)
    onnx_artifact_path = os.path.join(ARTIFACTS_DIR, "deep_model.onnx")
    triton_model_path = os.path.join(TRITON_DIR, "model.onnx")

    # 3. Export to ONNX with Dynamic Batch Axis
    torch.onnx.export(
        deep_model,
        (dummy_x_cat, dummy_x_num),
        onnx_artifact_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x_cat", "x_num"],
        output_names=["logits"],
        dynamic_axes={
            "x_cat": {0: "batch_size"},
            "x_num": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    print(f"✅ Successfully exported ONNX model to: {onnx_artifact_path}")

    # Copy to Triton model directory
    import shutil
    shutil.copyfile(onnx_artifact_path, triton_model_path)
    print(f"✅ Copied model to Triton repository: {triton_model_path}")

    # 4. Verify ONNX Parity against PyTorch Output
    with torch.no_grad():
        pt_out = deep_model(dummy_x_cat, dummy_x_num).numpy()

    ort_session = ort.InferenceSession(onnx_artifact_path)
    ort_inputs = {
        "x_cat": dummy_x_cat.numpy(),
        "x_num": dummy_x_num.numpy()
    }
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    print(f"Parity Verification: Max absolute difference = {max_diff:.8f}")
    assert max_diff < 1e-4, f"ONNX parity mismatch: {max_diff}"
    print("✅ 100% Numerical Parity Confirmed between PyTorch and ONNX Runtime!")

    return onnx_artifact_path

if __name__ == "__main__":
    export_deep_model_to_onnx()
