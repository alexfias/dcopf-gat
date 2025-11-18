# ML Demonstrator — Physics-Informed Graph Attention Network for Power Systems
This repository contains the core machine-learning model developed for the Destination Earth (DestinE) ML Demonstrator, a project integrating high-resolution climate data with advanced energy system modelling. The demonstrator showcases how physics-informed machine learning can accelerate Optimal Power Flow (OPF), power flow estimation, and scenario analysis under climate variability.

The model serves as the computational engine of the ML Demonstrator delivered under DestinE, linking climate-driven inputs with power system operation through a scalable, graph-based neural network.

## Model Architecture

The model is a Physics-Informed Graph Attention Network (PINN-GAT) based on the architecture described in:

Optimal Power Flow in a highly renewable power system based on attention neural networks
https://www.sciencedirect.com/science/article/abs/pii/S0306261924001624

The paper introduces the Spatial Multi-Window Graph Self-Attention Network (SMW-GSAT) and Node-Link Attention (NLAT) mechanisms, which directly inspire the attention structure, multi-hop processing, and node–edge embedding design used here.

Our implementation adapts these ideas for:

Surrogate DC-OPF and real-time flow mapping

Integration with climate-driven inputs (capacity factors, weather)

Physics-informed learning via nodal balance constraints

PCA-based dimensionality reduction

Deployment inside the DestinE ML Demonstrator stack

## Model Components

### ✔ Laplacian Positional Encodings (PE)
Encode electrical topology using eigenvectors of the normalized graph Laplacian.

### ✔ Multi-Window Graph Attention (SMW-GSAT)
Several stacked GAT layers with different hop sizes:

- Captures local + global electrical structure  
- Scales to larger grids  
- Mimics power flow locality and long-range interactions  

### ✔ Node–Link Attention (NLAT-inspired)
Constructs link embeddings using:

- Node embeddings  
- Node PE  
- Link PE (concatenated PE of endpoints)

### ✔ MLP Decoder
Maps link embeddings to predicted line flows or PCA components.

---

## Physics-Informed Loss

To ensure electrical feasibility, the model uses a **power-balance constraint**:

\[
g_i(t) \approx d_i(t) + \sum_\ell f_{\ell}(t)\, W_{\ell i}
\]

Loss components:

1. **Flow prediction** via log-cosh  
2. **Power-balance constraint** (MAE)  
3. **Optional PCA reconstruction**  
4. **Optional weighting of flow PCs**

This makes the model a **fast, physics-respecting surrogate** for full OPF calculations.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/alexfias/dcopf-gat.git
cd dcopf-gat
```

Create a Conda environment:

```bash
conda create -n dcopf-gat python=3.10
conda activate dcopf-gat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Example usage:

```python
from dcopf_gat.train import run_experiment

model, history, (test_x, test_y), test_metrics = run_experiment(
    data_dir="data_toy_3bus",
    learning_rate=1e-3,
    batch_size=32,
    epochs=50,
)
```

Outputs:

- A trained model  
- Training history  
- Test metrics  
- Predictions  

---

## Repository Structure

```
dcopf-gat/
│
├── dcopf_gat/                 # Core ML library
│   ├── data.py                # Data loading, preprocessing
│   ├── graph.py               # Graph + positional encoding
│   ├── model.py               # GAT layers + physics-informed loss
│   ├── train.py               # Training utilities
│   └── utils.py               # Helpers (normalization, seeding)
│
├── scripts/
│   ├── generate_toy_dataset.py  # Synthetic dataset generator
│   └── test_run.py              # Example model execution
│
├── configs/                   # Optional configuration files
├── experiments/               # Jupyter notebooks
├── data/                      # Placeholder for real datasets
├── README.md
└── requirements.txt
```

---

## Applications

This ML architecture enables:

- Fast surrogate DC-OPF  
- Real-time flow estimation  
- Adequacy analysis (ERAA-style)  
- Large-scale climate–energy scenario exploration  
- Uncertainty quantification workflows  
- Integration with DestinE Climate DT pipelines  

---

Reference

Please cite the foundational architecture if you use this model:
Optimal Power Flow in a highly renewable power system based on attention neural networks
https://www.sciencedirect.com/science/article/abs/pii/S0306261924001624

