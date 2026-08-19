import torch
from model import SimpleGRU
import pnnx

def export_to_ncnn():
    print("Loading GRU model...")
    model = SimpleGRU()
    
    # Load weights
    model.load_state_dict(torch.load("simple_gru.pth", weights_only=True))
    model.eval()

    # Generate dummy input
    dummy_input = torch.randn(1, 128, 24)

    print("Exporting model to NCNN via PNNX...")
    
    # PNNX intercepts the PyTorch model directly and generates native C++ NCNN operators
    # This outputs 'simple_gru_pnnx.ncnn.param' and 'simple_gru_pnnx.ncnn.bin'
    try:
        pnnx.export(
            model, 
            "simple_gru_pnnx.pt", 
            dummy_input
        )
        print("Export complete! Check for .param and .bin files in this directory.")
    except Exception as e:
        print("PNNX Python export failed. Falling back to TorchScript generation...")
        print(f"Error: {e}")
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save("simple_gru_traced.pt")
        print("\nSaved 'simple_gru_traced.pt'.")
        print("To convert it, run the PNNX binary manually in your terminal:")
        print("./pnnx simple_gru_traced.pt inputshape=[1,128,24]")

if __name__ == "__main__":
    export_to_ncnn()
