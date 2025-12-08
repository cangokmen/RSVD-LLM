<div align="center">
<h1>RSVD-LLM: Randomized SVD for Large Language Model Compression</h1>
</div>

## Introduction

This repository extends [SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM) with **Randomized SVD (RSVD)** implementation for significantly faster compression of large language models. 

### Original SVD-LLM Papers

> **[SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression](https://openreview.net/forum?id=LNYIUouhdt)**
> 
> *Xin Wang, Yu Zheng, Zhongwei Wan, Mi Zhang*   
> *The Ohio State University, Michigan State University*
> 
> International Conference on Learning Representations (ICLR) 2025

> **[SVD-LLM V2: Optimizing Singular Value Truncation for Large Language Model Compression](https://arxiv.org/abs/2503.12340)**
> 
> *Xin Wang, Samiul Alam, Zhongwei Wan, Hui Shen, Mi Zhang*  
> *The Ohio State University*
> 
> Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL) 2025

## Randomized SVD (RSVD) Implementation

**RSVD** provides significant speedup over standard SVD decomposition for large weight matrices while maintaining comparable model quality and perplexity.

### Key Benefits of RSVD

- **Faster Compression**: RSVD uses randomized algorithms to approximate SVD, providing **1.3x-1.4x speedup** for large matrices
- **Memory Efficient**: Computes only the top-k singular values/vectors needed for compression
- **Comparable Quality**: Maintains similar perplexity and model performance compared to standard SVD
- **Configurable**: Control the trade-off between speed and accuracy with `--rsvd_oversamples` and `--rsvd_n_iter` parameters

### Benchmark Results

Comprehensive benchmarks comparing RSVD vs SVD across compression ratios (0.1-0.5) are available in `benchmark_results_comparison/`. Visualizations can be generated using scripts in the `graphs/` folder:

- `perplexity_graph.py` - Model perplexity comparison
- `efficiency_graph.py` - Throughput comparison
- `compression_speedup_graph.py` - Compression time comparison
- `parameters_retained_graph.py` - Parameter retention analysis

## Quick Start

### Installation

**Important:** Keep the transformers package at exactly version 4.35.2, as the compressed model structure has modifications in the `component/` folder.

1. Create a conda environment with Python 3.9:
```bash
conda create -n compress python=3.9
conda activate compress
```

2. Clone the repository:
```bash
git clone https://github.com/cangokmen/RSVD-LLM.git
cd RSVD-LLM
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Examples

#### Standard SVD Compression
```bash
bash compress_llama.sh
```
Compresses LLaMA-7B with 20% compression ratio using standard SVD and evaluates perplexity and efficiency.

#### RSVD vs SVD Benchmark
```bash
# Set compression ratio (0.1 to 0.5)
export RATIO=0.2

# Run compression benchmark
bash compress_svd_vs_rsvd.sh

# Run evaluation benchmark
bash evaluate_svd_vs_rsvd.sh
```
This benchmarks both RSVD and SVD compression methods, measuring compression time, perplexity, and throughput.

    
## Usage

### RSVD Parameters

- `--rsvd_oversamples`: Number of additional samples for improved accuracy (default: 10)
- `--rsvd_n_iter`: Number of power iterations for better approximation (default: 2)

### Compression with RSVD

For compression ratios ≤ 0.3, use truncation-aware data whitening:

```bash
python SVDLLM.py \
  --step 1 \
  --ratio 0.2 \
  --model jeffwan/llama-7b-hf \
  --whitening_nsamples 256 \
  --dataset wikitext2 \
  --seed 0 \
  --model_seq_len 2048 \
  --save_path ./compressed_models \
  --rsvd_oversamples 10 \
  --rsvd_n_iter 2
```

**Faster compression:** Reduce `--rsvd_n_iter` to 1 for ~25% faster compression with minimal quality loss.


### 2. Parameter Update with Sequential Low-rank Approximation
We first update the compressed weight matrix U and then V with LoRA fine-tuning.
```
python LoRA.py \
--prune_model COMPRESSED_MODEL_PATH \
--data_path yahma/alpaca-cleaned \
--output_dir LORA_OUTPUT_PATH  \
--lora_r 8 \
--num_epochs 2 \
--learning_rate 1e-4 \
--batch_size 64
```

### 3. SVD-LLM + GPTQ
SVD-LLM can also be integrated with quantization methods to achieve a better compression. Here is the example of how to integrate SVD-LLM (20% compression ratio) with GPTQ-4bit to compress LLaMA-7B
```
bash svdllm_gptq.sh
```

### Evaluation

**Perplexity Evaluation:**
```bash
python SVDLLM.py \
  --step 4 \
  --model_path ./compressed_models/your_model.pt
```

Download the c4 dataset from [this link](https://drive.google.com/drive/folders/123Id1MkZVsKySGy_sMO4RgiJKrtPcvUp?usp=sharing) and place the JSON files in `utils/`.

**Efficiency Evaluation:**
```bash
python SVDLLM.py \
  --step 5 \
  --model_path ./compressed_models/your_model.pt
```

### Visualization

Generate comparison graphs from benchmark results:

```bash
cd graphs
python perplexity_graph.py
python efficiency_graph.py
python compression_speedup_graph.py
python parameters_retained_graph.py
```
## Repository Structure

```
RSVD-LLM/
├── SVDLLM.py                    # Main compression script
├── compress_svd_vs_rsvd.sh      # Benchmark RSVD vs SVD compression
├── evaluate_svd_vs_rsvd.sh      # Benchmark evaluation script
├── evaluater.py                 # Evaluation utilities
├── benchmark_results_comparison/ # Benchmark results (ratios 0.1-0.5)
├── graphs/                      # Visualization scripts
│   ├── perplexity_graph.py
│   ├── efficiency_graph.py
│   ├── compression_speedup_graph.py
│   └── parameters_retained_graph.py
├── component/                   # Modified model components
├── utils/                       # Utilities and RSVD implementation
│   └── rsvd.py                 # Randomized SVD implementation
└── gptq/                        # GPTQ integration
```

## Citation

If you use this work, please cite the original SVD-LLM papers:

```bibtex
@inproceedings{wang2025svdllm,
  title={{SVD}-{LLM}: Truncation-aware Singular Value Decomposition for Large Language Model Compression},
  author={Xin Wang and Yu Zheng and Zhongwei Wan and Mi Zhang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  url={https://openreview.net/forum?id=LNYIUouhdt}
}
```

## Acknowledgments

This repository is based on [SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM) by the AIoT-MLSys Lab at The Ohio State University.
