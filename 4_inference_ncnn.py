import ncnn
import numpy as np
import time
import os

def get_file_size_mb(filepath):
    if not os.path.exists(filepath): return 0
    return os.path.getsize(filepath) / (1024 * 1024)

def benchmark_ncnn(param_path, bin_path, iterations=100):
    if not os.path.exists(param_path) or not os.path.exists(bin_path):
        return 0

    net = ncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)

    input_array = np.random.rand(1, 128, 24).astype(np.float32)
    ncnn_mat = ncnn.Mat(input_array)

    # Warmup runs
    for _ in range(10):
        ex = net.create_extractor()
        ex.input("in0", ncnn_mat)
        _ = ex.extract("out0")
        
    # Benchmark runs
    start = time.time()
    for _ in range(iterations):
        ex = net.create_extractor()
        ex.input("in0", ncnn_mat) 
        _ = ex.extract("out0")
    
    return (time.time() - start) / iterations * 1000

def main():
    fp32_param = "simple_gru_pnnx.ncnn.param"
    fp32_bin = "simple_gru_pnnx.ncnn.bin"
    
    fp16_param = "simple_gru_fp16.ncnn.param"
    fp16_bin = "simple_gru_fp16.ncnn.bin"
    
    int8_param = "simple_gru_int8.ncnn.param"
    int8_bin = "simple_gru_int8.ncnn.bin"
    
    print("\n==============================")
    print("   NCNN Model Size Comparison ")
    print("==============================")
    
    fp32_size = get_file_size_mb(fp32_param) + get_file_size_mb(fp32_bin)
    print(f"NCNN (FP32):         {fp32_size:.2f} MB")
    
    fp16_size = get_file_size_mb(fp16_param) + get_file_size_mb(fp16_bin)
    if fp16_size > 0: print(f"NCNN (FP16):         {fp16_size:.2f} MB")
        
    int8_size = get_file_size_mb(int8_param) + get_file_size_mb(int8_bin)
    if int8_size > 0: print(f"NCNN (INT8):         {int8_size:.2f} MB")

    print("\n==============================")
    print(" NCNN Inference Speed (CPU)   ")
    print("==============================")
    
    fp32_time = benchmark_ncnn(fp32_param, fp32_bin)
    print(f"NCNN (FP32):         {fp32_time:.2f} ms/iter")
    
    if fp16_size > 0:
        fp16_time = benchmark_ncnn(fp16_param, fp16_bin)
        print(f"NCNN (FP16):         {fp16_time:.2f} ms/iter")
        
    if int8_size > 0:
        int8_time = benchmark_ncnn(int8_param, int8_bin)
        print(f"NCNN (INT8):         {int8_time:.2f} ms/iter")

if __name__ == "__main__":
    main()
