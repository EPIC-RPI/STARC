# STARC: Selective Token Access with Remapping and Clustering for Efficient LLM Decoding on PIM Systems

This repository provides a complete workflow to reproduce the key results of **STARC**, including:

1. The implementation of STARC’s selective token access with **KV remapping** and **online clustering**.  
2. Evaluation scripts to reproduce:
   - accuracy results on **LongBench** and **RULER**, and
   - perplexity results on **PG-19**.  
3. The simulator setup to reproduce **system-level performance/energy** results on GPU–PIM platforms based on the **AttAcc** simulator (Ramulator-based).

---

## Contents

- [What’s in this artifact](#whats-in-this-artifact)
- [Requirements](#requirements)
  - [Hardware](#hardware)
  - [Software](#software)
  - [Resource estimate](#resource-estimate)
- [Getting started](#getting-started)
  - [1) Get the code](#1-get-the-code)
  - [2) Create the environment](#2-create-the-environment)
  - [3) Set up the PIM system simulator](#3-set-up-the-pim-system-simulator)
  - [4) Build Ramulator2](#4-build-ramulator2)
- [Reproducing paper results](#reproducing-paper-results)
  - [E1: LongBench accuracy](#e1-longbench-accuracy)
  - [E2: PG-19 perplexity](#e2-pg-19-perplexity)
  - [E3: RULER (32K context)](#e3-ruler-32k-context)
  - [E4: GPU–PIM system simulation](#e4-gpupim-system-simulation)
- [Outputs](#outputs)
- [Methodology](#methodology)
- [License](#license)

---

## What’s in this artifact

- **Algorithm:** The STARC algorithm, which enables efficient long-context LLM inference by selectively accessing and remapping KV cache entries via online clustering under a fixed KV-cache budget.
- **Program:** The STARC artifact running public long-context benchmarks: LongBench (16 datasets) and RULER (13 datasets).
- **Models:** LongChat-7B v1.5-32K; LLaMA-3.1-8B-Instruct; Mistral-7B-Instruct-v0.3 (publicly available via Hugging Face).
- **Datasets:** LongBench (16 datasets; e.g., HotpotQA, QASPER, GovReport, etc.); PG-19; RULER (13 datasets; e.g., NIAH Single, Multi-key NIAH, Multi-value NIAH, etc.), all publicly available (Hugging Face).
- **Metrics:** LongBench task scores; PG-19 perplexity; RULER task scores; and system metrics such as latency and energy.
- **Outputs:** LongBench/RULER scores, PG-19 perplexity traces, and system-level performance/energy metrics with breakdowns.
- **Availability:** Publicly available.
- **License:** MIT license.

---

## Requirements

### Hardware

- **LLM accuracy evaluation (LongBench / PG-19 / RULER):** Compatible with commonly used NVIDIA GPUs. We recommend NVIDIA **H100** or **L40** with sufficient GPU memory (e.g., at least **48 GB per GPU**).
- **System-level simulation:** CPU-only execution is sufficient. Experiments in the paper were conducted on a dual-socket AMD EPYC 9334 system with 64 CPU cores in total (2×32 cores).

### Software

- **Python:** 3.10  
- **CUDA:** 12.8  
- **Python dependencies:** see `pyproject.toml`

### Resource estimate

- **Disk space:** ~80 GB total  
- **Setup time:** ~20 minutes  
- **Experiment time:**  
  - Model accuracy experiments: ~12 hours (excluding additional appendix results)  
  - System-level performance experiments: ~24 hours  

---

## Getting started

### 1) Get the code

```bash
git clone --recurse-submodules https://github.com/EPIC-RPI/STARC
cd STARC
```

### 2) Create the environment

To better reproduce the results and avoid potential conflicts, we recommend using Python 3.10 and CUDA 12.8.

We provide scripts for the recommended environment setup. Please follow the instructions below to create the conda environment and install the STARC packages:

```bash
conda create -yn STARC python=3.10
conda activate STARC
pip install ninja==1.11.1.1 packaging
pip install -e .
pip install flash-attn==2.3.0 --no-build-isolation
conda install -c conda-forge cupy
conda install numpy scikit-learn
conda install datasets
conda install cmake
```

### 3) Set up the PIM system simulator

This artifact builds on the AttAcc simulator:

```bash
cd simulator_starc
git submodule update --init --recursive
```

### 4) Build Ramulator2

```bash
bash set_pim_ramulator.sh
cd ramulator2
mkdir build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j
cp ramulator2 ../ramulator2
cd ../../
```

---

## Reproducing paper results

This section describes how to reproduce the key results reported in the paper.

### E1: LongBench accuracy

To reproduce the LongBench accuracy results:

```bash
cd <Your Path>/STARC/scripts/
sh longbench.sh
```

If you want to evaluate more models, the corresponding model paths are defined in:

```text
STARC/evaluation/LongBench/config/model2path.json
```


By replacing the model name in `longbench.sh`, you can evaluate STARC under different models reported in the paper.

### E2: PG-19 perplexity

To reproduce the perplexity results on PG-19:

```bash
cd <Your Path>/STARC/scripts/
sh ppl_eval.sh
```

### E3: RULER (32K context)

To reproduce RULER results on **LLaMA-3.1-8B-Instruct**, the RULER testing data are already included in the `STARC/ruler` directory.

To reproduce the RULER results under a 32K context length:

```bash
cd <Your Path>/STARC/scripts/
sh RULER.sh
```

### E4: GPU–PIM system simulation

The system-level simulation experiments are conducted using the AttAcc-based simulator.

#### Full attention

To reproduce the results for full attention:

```bash
python main.py --system dgx-attacc --gpu H100 --ngpu 8 --model Mistral-7B \
  --lin 2048 --lout 24000 --batch 16 --pim bank \
  --powerlimit --ffopt --pipeopt
```

#### Sparse attention configurations

To reproduce the results for configurations with sparse attention methods:

```bash
python main.py --system dgx-attacc --gpu H100 --ngpu 8 --model Mistral-7B \
  --lin 2048 --lout 24000 --batch 16 --pim bank \
  --powerlimit --ffopt --pipeopt \
  --sparsity --kv_budget_table kv_budget_Mistral_STARC.txt
```

Different sparse attention methods and models use different `.txt` files specified by the `--kv_budget_table` option. These files are derived from the attention masks produced by each method at each decoding step in real inference tasks (e.g., LongBench), and map them to the row-level granularity of the PIM architecture, where each DRAM row activation fetches 16 key/value vectors in parallel. They define how many memory rows are activated at each decoding step and are used to guide the simulator accordingly.

> **Note 1:** When switching the evaluated method (Full attention / STARC / SparQ / Quest), please delete the previously generated `ramulator.out`; otherwise, cached results from the last run may be reused.

> **Note 2:** When reproducing methods other than STARC, comment out the following two lines in `STARC/simulator_starc/src/ramulator_wrapper.py` to avoid introducing clustering overhead:
> ```python
> if l == l_target - 1:
>     trace_args += " --add_cluster"
> ```



---

## Outputs

- **Model accuracy experiments:** Each `.sh` script generates a corresponding `.jsonl` file for each model and each task. These files contain the ground-truth answers, model predictions, and evaluation scores.
- **PG-19 perplexity:** A `.txt` file is generated to record the evolution of perplexity during evaluation.
- **Simulation experiments:** The simulator produces `.xlsx` or `.csv` files that record the breakdown of end-to-end latency and energy.

---

## License

This project is released under the **MIT License**.

---

## Acknowledgements

This repository incorporates and builds upon open-source implementations from the following projects. We thank the authors for making their code publicly available.

- **Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference**  
  Codebase: https://github.com/mit-han-lab/quest

- **AttAcc! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference**  
  Simulator: https://github.com/scale-snu/attacc_simulator
