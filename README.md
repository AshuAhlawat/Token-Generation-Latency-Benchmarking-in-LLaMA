# Token Generation Latency Benchmarking in LLaMA

### Objective
The goal of this project is to measure per-token latency and throughput of the LLaMA model at varying batch sizes {1, 8, 32, 128}. The deliverables will be latency vs. throughput graphs and a scaling analysis that explains how efficiency changes with batch size.

### Background
In large language models, latency (time to generate each token) and throughput (tokens per second) are critical for both interactive and high-throughput applications. Measuring how these metrics scale with batch size may demonstrates system bottlenecks such as GPU utilization, memory usage, and attention mechanisms. Specially in Transformer arcitecture models in which system configuration is even more key, so we want the best possible estimates of the computation we can undergo within a realistic timeframe.

### Proposed Methodology

1. Implementation
    - Use the Ollama library to run the LLaMA model.
    - Enable streaming output to capture per-token (or per-chunk) timings.
    - Log timestamps for each token generated and compute per-token latency.
    - Run experiments for batch sizes {1, 8, 32, 128} (simulated via concurrent requests).

2. Measurement Practices
    - Perform warm-up runs to avoid startup overhead.
    - Collect multiple iterations per configuration to compute averages and variance.
    - Synchronize timing to ensure correctness.
    - Run the measurements on different batchs with different complications of prompts.
    - Vary generation length (e.g., 16, 64, 256) to see dependency on sequence length.
    - Repeat with different precisions (fp32 vs fp16 vs int8) to show speed/accuracy tradeoffs.

### Analysis

- Plot latency vs. batch size and throughput vs. batch size.
- A scaling analysis explaining performance trade-offs.
- Judging if the inherent difficulty of a question make the model process using more computation or time

### System Config

- Modules
    - Ollama for replicating a local copy of llama3 to test
    - pytorch for any device configurations
    - matplotlib for plots
    - Beautifulsoup4 , Selenium for scraping data

- GPU Specifications

```
C:\Users\ashua>nvidia-smi  
Sat Sep 13 09:27:11 2025  
+-----------------------------------------------------------------------------------------+  
| NVIDIA-SMI 581.08                 Driver Version: 581.08         CUDA Version: 13.0     |  
+-----------------------------------------+------------------------+----------------------+  
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |  
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |  
|                                         |                        |               MIG M. |  
|=========================================+========================+======================|  
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |  
| N/A   43C    P8              2W /  135W |       0MiB /   8188MiB |      0%      Default |  
|                                         |                        |                  N/A |  
+-----------------------------------------+------------------------+----------------------+  
```
- Controlling CPU threads and context length  
    `options={"num_thread": 4, "num_ctx": 2048}`


### System Logging
    `ollama ps`