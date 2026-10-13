# TRACE: Topological Relativistic Analysis of Chaotic Emission 🌌

[![Julia 1.9+](https://img.shields.io/badge/Julia-1.9%2B-9558B2.svg)](https://julialang.org/)
[![CUDA Ready](https://img.shields.io/badge/CUDA-Enabled-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TRACE** is a high-performance computational framework for analyzing the post-Newtonian (1PN) restricted three-body problem. It leverages massively parallel GPU computing to simulate gravitational wave emissions and map ultra-high-resolution fractal basins of attraction (Wada property) in critically chaotic relativistic systems.

## 📌 Features

* **Relativistic Dynamics:** Integrates the Einstein-Infeld-Hoffmann (EIH) equations of motion at the 1PN approximation.
* **Gravitational Waves:** Extracts cross-polarized strain ($h_+$) from the quadrupolar emission of highly eccentric, chaotic orbits.
* **Ultra-HD Fractal Topology:** Generates dense topological maps (up to $10,000 \times 10,000$ grids) to compute the Wada parameter ($W$) and uncertainty exponent ($\alpha$) using CUDA acceleration.
* **Publication-Ready Visualization:** Automated Python pipelines to render dark-academic and high-contrast Q1 journal figures.

## 🚀 Prerequisites

To run the simulations and generate the figures, you need a Linux environment configured with the following dependencies:

### 1. Julia Environment (Simulation Engine)
* [Julia](https://julialang.org/downloads/) 
* `CUDA.jl` (Requires an NVIDIA GPU)
* `StaticArrays.jl`
* `CairoMakie.jl` (For statistical plotting)

### 2. Python Environment (Visualization)
* Python 3.8+
* `numpy`
* `matplotlib`

## 🛠️ Pipeline Execution

The entire workflow—from data generation on the GPU to the final rendering of the Q1 manuscript panels—is automated via a bash script.

To execute the full pipeline:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
