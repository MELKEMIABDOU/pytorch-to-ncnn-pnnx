import torch
import torch.nn as nn

class SimpleGRU(nn.Module):
    def __init__(self, input_size=24, hidden_size=64, num_layers=2, output_size=2):
        super(SimpleGRU, self).__init__()
        # Configure input shape as (batch_size, sequence_length, features)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Expected input shape: (batch_size, sequence_length, input_size)
        out, _ = self.gru(x)
        
        # Extract output from the final time step
        final_out = self.fc(out[:, -1, :])
        return final_out
