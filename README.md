# PyTorch to NCNN Export (The PNNX Approach)

This repository demonstrates how to export a PyTorch Recurrent Neural Network (GRU) directly into the **NCNN** format, bypass ONNX entirely, and apply **FP16** and **INT8** quantization.

##  Why avoid ONNX here? (The Standard vs. PNNX Method)

### 1. The Standard Way (ONNX to NCNN)
For most standard models (like CNNs used for image recognition), the normal pipeline is:
`PyTorch -> ONNX -> NCNN (using the onnx2ncnn tool)`.
This works flawlessly for linear networks because ONNX represents convolutions very well.

### 2. The GRU Way (PyTorch to PNNX to NCNN)
When dealing with **Recurrent Neural Networks (RNNs, LSTMs, GRUs)**, the standard ONNX pipeline frequently fails. ONNX uses complex `Loop` and `Scan` operators to represent time-series sequences, which `onnx2ncnn` struggles to compile into native C++ code.

**The Solution:** We use **PNNX (PyTorch Neural Network Exchange)**. 
PNNX intercepts the PyTorch TorchScript JIT graph and seamlessly fuses PyTorch GRU cells directly into their native C++ NCNN equivalents, bypassing ONNX entirely and preventing corrupted graphs.

##  Project Structure

* `model.py`: Defines the PyTorch `SimpleGRU` architecture (simulating a 24-feature, 128-sequence telemetry model).
* `1_train_model.py`: Initializes the GRU and saves the `.pth` weights.
* `2_export_pnnx.py`: Loads the `.pth` weights and uses `pnnx.export()` to generate the native FP32 NCNN `.param` (topology) and `.bin` (weights) files.
* `3_quantize_model.py`: Applies PyTorch-level quantization to the GRU, shrinking it to **FP16** and **INT8**, and then exports those directly to NCNN via PNNX.
* `4_inference_ncnn.py`: Loads the raw NCNN files into memory and benchmarks the file sizes and CPU inference speeds for all variants (FP32, FP16, INT8).

## 🛠️ Setup & Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

##  How to run

Run the steps in order:

1. **Initialize the model**
   ```bash
   python 1_train_model.py
   ```
2. **Export to NCNN directly (FP32)**
   ```bash
   python 2_export_pnnx.py
   ```
3. **Generate FP16 and INT8 NCNN models**
   ```bash
   python 3_quantize_model.py
   ```
4. **Run NCNN Inference Benchmark**
   ```bash
   python 4_inference_ncnn.py
   ```
