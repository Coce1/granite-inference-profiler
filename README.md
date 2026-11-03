# Granite Inference Profiler

## Overview
This repository contains a benchmarking and profiling tool for LLM inference on the GPU, built around IBM's **Granite** models (`transformers` + PyTorch). It measures what actually matters when serving a language model: the latency of the first token, the cost of each generated token, the throughput, and the VRAM footprint.

Timing a GPU program is easy to get wrong because CUDA kernels are asynchronous. The tool synchronizes the GPU before and after every timed section, warms up with the exact configuration of the measurement, repeats the run several times and reports median, mean, standard deviation, minimum and maximum. Results are exported as JSON and CSV, plotted with matplotlib, and an optional `torch.profiler` trace can be generated for deeper analysis.

## Project Structure
* **`src/Granite_Inference_Pipeline.py`**: The whole pipeline (CLI, model loading, timed generation, statistics, export, optional profiling).
* **`tests/test_pipeline.py`**: `pytest` suite (unit tests with a fake model, integration tests on a tiny Hugging Face model, GPU tests, and an optional test on the real Granite model).
* **`pytest.ini`**: Test configuration and markers (`integration`, `gpu`, `slow`).
* **`requirements.txt`** / **`requirements-dev.txt`**: Runtime dependencies / runtime + `pytest`.
* **`Makefile`**: Virtual environment, install, run, profile, clean.
* **`.gitignore`** / **`.gitattributes`**: Ignored results and traces, normalized line endings (the `Makefile` must stay in LF).

## Measured Metrics
| Metric | Definition |
| :--- | :--- |
| **TTFT** (ms) | Time to first token: a separate 1-token generation (prefill + first token) |
| **Total latency** (ms) | Full generation of `--max-new-tokens` tokens |
| **Decode latency** (ms/token) | `(total - TTFT) / (new_tokens - 1)` |
| **Decode throughput** (tokens/s) | `batch_size × (new_tokens - 1) / (total - TTFT)` |
| **End-to-end throughput** (tokens/s) | `batch_size × new_tokens / total` |
| **Model VRAM** (MB) | `torch.cuda.memory_allocated()` growth after loading the model |
| **Peak VRAM** (MB) | `torch.cuda.max_memory_allocated()`, reset right before the timed generation |

## Prerequisites
* **Python** 3.10+.
* **NVIDIA GPU** recommended (CUDA build of PyTorch). The tool also runs on CPU, but very slowly.
* **Hugging Face access** to the model (`ibm-granite/granite-3.0-2b-instruct` by default, about 5 GB). Setting `HF_TOKEN` is optional and only raises download rate limits.
* **Precision:** `--dtype auto` picks `bfloat16` on Ampere or newer (compute capability ≥ 8), `float16` on older GPUs such as the T4, and `float32` on CPU.

## Build and Execute
```bash
# Clone the repository
git clone https://github.com/Coce1/granite-inference-profiler.git
cd granite-inference-profiler

# Create the virtual environment and install the dependencies
make install

# Run the benchmark (defaults: 2 warm-ups, 5 measured runs, 256 new tokens)
make run

# Pass any option through ARGS
make run ARGS="--dtype float16 --repeats 10 --batch-size 4 --max-new-tokens 128"

# Same run with a torch.profiler trace
make profile
```

On Google Colab, PyTorch is already installed:
```bash
!git clone https://github.com/Coce1/granite-inference-profiler.git
%cd granite-inference-profiler
!pip install -q -r requirements-dev.txt
!python src/Granite_Inference_Pipeline.py --repeats 5 --max-new-tokens 64
```

### Command-line options
| Option | Default | Description |
| :--- | :--- | :--- |
| `--model` | `ibm-granite/granite-3.0-2b-instruct` | Hugging Face id or local path |
| `--device` | `auto` | `auto`, `cuda` or `cpu` |
| `--dtype` | `auto` | `auto`, `float16`, `bfloat16` or `float32` |
| `--prompt` / `--prompt-file` | built-in prompt | Prompt text, or a text file (takes priority) |
| `--no-chat-template` | off | Send the raw prompt instead of using `apply_chat_template` |
| `--max-new-tokens` | 256 | Generated tokens per sequence |
| `--batch-size` | 1 | Sequences generated in parallel |
| `--warmup` / `--repeats` | 2 / 5 | Warm-up generations / measured runs |
| `--allow-early-stop` | off | Allow stopping on EOS (otherwise the length is fixed so that runs are comparable) |
| `--seed` | 42 | Random seed |
| `--output-dir` | `results` | Output directory |
| `--no-plots` | off | Skip the matplotlib figure |
| `--profile` / `--profile-tokens` | off / 32 | Export a `torch.profiler` trace over a short generation |

Exit codes: `0` success, `2` CUDA out of memory, `3` model loading error.

### Output files
Each run writes to `results/<timestamp>_<model>_<dtype>/`:
* **`results.json`**: metadata (GPU, CUDA, `torch` and `transformers` versions, all options), summary statistics and every run.
* **`runs.csv`**: one row per measured run.
* **`benchmark.png`**: TTFT, decode latency and decode throughput per run.
* **`trace.json`** (with `--profile`): Chrome trace, to open in `chrome://tracing` or `ui.perfetto.dev`.

## Benchmark Protocol
1. **Load** the model and tokenizer (left padding, required for batched generation). The prompt goes through the model's **chat template**, since Granite is an instruct model.
2. **Warm-up** (2 generations) with the same configuration as the measurement.
3. **Timing** with `time.perf_counter()` and `torch.cuda.synchronize()` before and after each timed section. Without synchronization, asynchronous kernels would make the GPU look much faster than it is.
4. **Fixed generation length** (`min_new_tokens = max_new_tokens`) so that every run generates the same number of tokens.
5. **Peak VRAM** reset with `torch.cuda.reset_peak_memory_stats()` right before the timed generation, so that the loading peak is not counted.
6. **Statistics** over the measured runs: median, mean, standard deviation, min, max.
7. **Cleanup**: references are dropped, then `gc.collect()` and `torch.cuda.empty_cache()`, so several models can be compared in the same process.

## Tests
```bash
pip install -r requirements-dev.txt

# Fast tests: fake model, no GPU, no download
pytest tests -v -m "not integration and not gpu and not slow"

# Integration (tiny Hugging Face model) + GPU tests
pytest tests -v -m "integration or gpu"

# Real Granite model on a GPU (about 5 GB download)
RUN_GRANITE=1 pytest tests -v -m slow
```
The real-model test also checks that the VRAM is released at the end of the run.

## Performance Analysis

**Hardware Environment:** Google Colab | GPU: Tesla T4 (sm_75, 15 GB) | Python 3.13
**Configuration:** `ibm-granite/granite-3.0-2b-instruct`, batch 1, prompt of 48 tokens, 64 generated tokens, 3 measured runs after 2 warm-ups. Values are medians.

| Precision | TTFT (ms) | Decode (ms/token) | Decode (tokens/s) | Model VRAM (MB) | Peak VRAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **float16** | **62.2** | 49.1 | 20.4 | 4832 | 4853 |
| **bfloat16** (emulated on T4) | 142.9 | 49.0 | 20.4 | 4832 | 4853 |
| **float32** | 113.8 | 49.3 | 20.3 | 9665 | 9693 |

### Key Takeaways

*   **float16 is the right choice on a T4 (TTFT 2.3x lower than bfloat16).** The T4 (compute capability 7.5) has no native bfloat16 units: PyTorch reports bfloat16 as "supported" but emulates it, which makes the prefill much slower. `--dtype auto` therefore checks the compute capability instead of `torch.cuda.is_bf16_supported()`.
*   **Decode speed does not depend on the precision (about 49 ms/token in all three cases).** float32 reads twice as many bytes per token as float16 (9.7 GB against 4.8 GB), yet it is not slower. For a memory-bound decode on a 320 GB/s GPU, one would expect about 15 ms/token in float16 and about 30 ms/token in float32. The measured 49 ms is therefore much higher and constant, which suggests that decoding is limited by per-token overhead (Python and kernel launches in `generate`) rather than by GPU memory bandwidth. This explanation is consistent with the numbers but has not been confirmed with a profiler: the next steps are `--profile` and a `--batch-size` sweep (if the overhead dominates, throughput should grow almost linearly with the batch size).
*   **TTFT contains one step of that overhead.** The TTFT of 62 ms in float16 is only slightly above one decode step (49 ms), so the prefill itself is probably short; this is an estimate, not a measurement.
*   **VRAM:** float32 needs exactly twice the weights memory of float16 (9665 MB against 4832 MB). Generation at batch 1 adds only about 21 MB (float16) to 28 MB (float32) on top of the weights.
*   **Variability and reproducibility:** two separate Colab sessions gave the same bfloat16 TTFT (142.9 and 142.8 ms) but decode medians of 20.4 and 20.0 tokens/s, with a standard deviation between 1.9 and 2.3 tokens/s over 3 runs. Colab is a shared machine: use `--repeats 10` or more before comparing two configurations.

### Correction of the previous version
The first version of the script timed a single generation with `time.time()` and no GPU synchronization, reported one number without any statistic, mixed the loading peak with the inference peak in its VRAM figure, forced float16 even on CPU, and relied on a `dtype` argument that older `transformers` versions do not accept. `torch` was also missing from `requirements.txt`. All of these are fixed in the current version.
