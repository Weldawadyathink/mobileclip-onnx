import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType

parser = argparse.ArgumentParser(description="Quantize an ONNX model using ONNX Runtime dynamic quantization.")
parser.add_argument('--input', type=str, required=True, help='Path to the input ONNX model')
parser.add_argument('--output', type=str, required=True, help='Path to save the quantized ONNX model')
args = parser.parse_args()

input_model_path = args.input
output_model_path = args.output

# Perform dynamic quantization
quantize_dynamic(
    model_input=input_model_path,
    model_output=output_model_path,
    weight_type=QuantType.QInt8,
)

print(f"Quantized model saved to {output_model_path}") 