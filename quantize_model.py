import torch
from model import SimpleGRU
import pnnx
import os

def quantize_fp16():
    print("Loading and converting PyTorch GRU to FP16...")
    model = SimpleGRU()
    model.load_state_dict(torch.load("simple_gru.pth", weights_only=True))
    
    # Convert model weights to FP16
    model.half()
    model.eval()

    # Create FP16 dummy input
    dummy_input = torch.randn(1, 128, 24).half()

    print("Exporting FP16 model to NCNN via PNNX...")
    try:
        pnnx.export(model, "simple_gru_fp16.pt", dummy_input)
        print("FP16 Export complete! (simple_gru_fp16.ncnn.param & .bin)")
    except Exception as e:
        print(f"FP16 Export failed: {e}")
        print("Using torch.jit.trace fallback...")
        traced = torch.jit.trace(model, dummy_input)
        traced.save("simple_gru_fp16.pt")

def quantize_int8():
    print("Loading and dynamically quantizing PyTorch GRU to INT8...")
    model = SimpleGRU()
    model.load_state_dict(torch.load("simple_gru.pth", weights_only=True))
    
    # Quantize PyTorch model dynamically to INT8
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.GRU, torch.nn.Linear}, dtype=torch.qint8
    )
    quantized_model.eval()

    # Create standard FP32 input (Dynamic INT8 models take FP32 inputs)
    dummy_input = torch.randn(1, 128, 24)

    print("Exporting INT8 model to NCNN via PNNX...")
    try:
        pnnx.export(quantized_model, "simple_gru_int8.pt", dummy_input)
        print("INT8 Export complete! (simple_gru_int8.ncnn.param & .bin)")
    except Exception as e:
        print("\nNote: Direct INT8 export via PNNX Python API can sometimes fail if operations are unsupported.")
        print(f"Error: {e}")
        print("For production NCNN INT8, it is recommended to use the NCNN C++ 'ncnn2table' tool with a calibration dataset (Static Quantization), exactly like you did with the Qualcomm QNN tools!")

if __name__ == "__main__":
    if not os.path.exists("simple_gru.pth"):
        print("Error: simple_gru.pth not found. Run step 1 first.")
    else:
        print("--- Starting NCNN Quantization ---")
        quantize_fp16()
        print("-" * 30)
        quantize_int8()
