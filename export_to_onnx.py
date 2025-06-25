import os
import torch
from mobileclip import create_model_and_transforms
import argparse

parser = argparse.ArgumentParser(description="Export MobileCLIP model to ONNX.")
parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (.pt file)')
parser.add_argument('--onnx', type=str, required=True, help='Path to save the exported ONNX file')
args = parser.parse_args()

CHECKPOINT_PATH = args.checkpoint
ONNX_PATH = args.onnx
DEVICE = "cpu"

# Derive model name from checkpoint path
model_name = os.path.splitext(os.path.basename(CHECKPOINT_PATH))[0]
if model_name == "mobileclip_blt":
    model_name = "mobileclip_b"

# Load model
model, _, preprocess = create_model_and_transforms(
    model_name=model_name,
    pretrained=CHECKPOINT_PATH,
    reparameterize=True,
    device=DEVICE,
)

# Set to eval mode
model.eval()

# Prepare dummy inputs
# Image: batch_size x 3 x 256 x 256 (from config)
dummy_image = torch.randn(1, 3, 256, 256, device=DEVICE)
# Text: batch_size x context_length (from config, context_length=77)
dummy_text = torch.randint(0, 49408, (1, 77), device=DEVICE)

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_image, dummy_text),
    ONNX_PATH,
    input_names=["image", "text"],
    output_names=["image_features", "text_features", "logit_scale"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "text": {0: "batch_size"},
    },
    opset_version=17,  # You can adjust this if needed
)

print(f"Model exported to {ONNX_PATH}")
