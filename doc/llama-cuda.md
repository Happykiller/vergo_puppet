# Building `llama-cpp-python` with CUDA

This guide explains how to compile `llama-cpp-python` with CUDA acceleration so that models can leverage your NVIDIA GPU. It is adapted from the long instructions previously included in the main README.

## Prerequisites

- An NVIDIA GPU with recent drivers installed
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Python 3.8 or newer
- Build tools such as `cmake`, `ninja`, and development headers

Install system packages on Ubuntu/Debian:

```bash
sudo apt update && sudo apt install -y \
  build-essential cmake ninja-build \
  python3-dev python3-pip \
  libopenblas-dev libsqlite3-dev libssl-dev
```

## Compilation Steps

1. **Clone the repository**
   ```bash
git clone https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
```
2. **Build with CUDA enabled**
   ```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```
   To force a full rebuild:
   ```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

### Option Breakdown

| Option            | Description                                  |
| ----------------- | -------------------------------------------- |
| `GGML_CUDA=on`    | Enable CUDA backend                           |
| `LLAMA_CUBLAS=on` | (Optional) Use cuBLAS for best performance    |
| `FORCE_CMAKE=1`   | Force native compilation (skip wheels)        |
| `--no-cache-dir`  | Avoid using cached builds                     |

## Validate the Installation

Check that CUDA support is available:

```bash
python3 -c "import llama_cpp; print(hasattr(llama_cpp, 'llama_create_context_with_cuda'))"
# Should print: True
```

## Using Locally Built Version in `requirements.txt`

If you want your project to use the locally compiled version:

```text
llama-cpp-python @ file:///path/to/llama-cpp-python
```

Example:

```text
llama-cpp-python @ file:///mnt/data/projects/llama-cpp-python
```

## Runtime Output Example

When successfully running, CUDA initialization will look like this:

```bash
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4080
...
load_tensors: layer 12 assigned to device CUDA0
```

## Optional: CUDA Setup in WSL2

If you are using WSL2, you may install the CUDA toolkit as follows:

```bash
sudo apt update
sudo apt install -y nvidia-cuda-toolkit
```

Verify `nvcc` is available:

```bash
nvcc --version
```

Monitor GPU usage:

```bash
watch -n 1 nvidia-smi
```

## Download the Devstral GGUF Model

The `usecase_summarize_mr` feature relies on a quantized GGUF model compatible with LLaMA.

1. Create the `models/` directory if needed:
   ```bash
mkdir -p models
```
2. Download the Devstral model:
   ```bash
huggingface-cli download "mistralai/Devstral-Small-2505_gguf" --include "devstralQ4_K_M.gguf" --local-dir "./models"
```
   Update the URL if your model is hosted elsewhere.
