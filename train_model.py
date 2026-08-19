import torch
from model import SimpleGRU

def main():
    print("Initializing untrained SimpleGRU model...")
    
    # Initialize GRU model with random weights
    model = SimpleGRU()
    model.eval()

    # Save PyTorch model weights
    model_path = "simple_gru.pth"
    torch.save(model.state_dict(), model_path)
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
