# ML Demonstrator — Physics-Informed Graph Attention Network for Power Systems

This repository contains the core machine-learning model developed for the **Destination Earth (DestinE) ML Demonstrator**, integrating high-resolution climate data with advanced energy system modelling.

The demonstrator showcases how **physics-informed machine learning** can accelerate:

- Optimal Power Flow (OPF)
- Power flow estimation
- Large-scale scenario analysis under climate variability

The model serves as the computational engine of the DestinE ML Demonstrator, linking climate-driven inputs with power system operation through a **scalable, graph-based neural network**.

---

## Model Architecture

The model is a **Physics-Informed Graph Attention Network (PINN-GAT)** inspired by:

> *Optimal Power Flow in a highly renewable power system based on attention neural networks*  
> https://www.sciencedirect.com/science/article/abs/pii/S0306261924001624

The paper introduces:
- **Spatial Multi-Window Graph Self-Attention (SMW-GSAT)**
- **Node–Link Attention (NLAT)**

These concepts directly inform the attention structure, multi-hop processing, and node–edge embedding design used here.

Our implementation adapts these ideas for:

- Surrogate **DC power flow and DC-OPF**
- Climate-driven inputs (capacity factors, weather)
- Physics-informed learning via nodal balance constraints
- PCA-based dimensionality reduction
- Deployment inside the DestinE ML Demonstrator stack

---

## Model Components

### Laplacian Positional Encodings (PE)
Encode electrical topology using eigenvectors of the normalized graph Laplacian.

### Multi-Window Graph Attention (SMW-GSAT)
Stacked graph-attention layers with multiple hop sizes:
- Capture local and global electrical structure  
- Scale to larger transmission grids  
- Reflect power flow locality and long-range interactions  

### Node–Link Attention (NLAT-inspired)
Constructs link embeddings from:
- Node embeddings  
- Node positional encodings  
- Link positional encodings (concatenated endpoint PEs)

### MLP Decoder
Maps link embeddings to predicted line flows (or PCA components).

---

## Physics-Informed Loss

To enforce electrical feasibility, the model includes a **power-balance constraint**:

\[
g_i(t) \approx d_i(t) + \sum_\ell f_{\ell}(t)\, W_{\ell i}
\]

Loss components:
1. Flow prediction loss (log-cosh)
2. Power-balance constraint (MAE)
3. Optional PCA reconstruction
4. Optional weighting of flow PCs

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

Install the package in editable mode (recommended):
```bash
pip install -e .
```

## Quick Start (Toy Dataset)

This repository includes a synthetic IEEE-14 DC power flow / DC-OPF dataset generator.

### 1. Generate the IEEE-14 Network dataset

From the repository root:

```bash
python make_toy_ieee14.py
```
This creates
```bash
data_toy_ieee14/
```

## Training the Model



```python
python -m scripts.run_experiment \
  --data_dir data_ieee14 \
  --batch_size 256 \
  --epochs 50

```
The training script automatically:

uses GPU if available

falls back to CPU otherwise


GPU mixed precision
```python
python -m scripts.run_experiment \
  --data_dir data_ieee14 \
  --device gpu \
  --mp \
  --batch_size 256 \
  --epochs 50
```

Programmatic Usage
```python
from dcopf_gat.train import run_experiment

model, history, (test_x, test_y), test_metrics = run_experiment(
    data_dir="data_ieee14",
    learning_rate=1e-3,
    batch_size=256,
    epochs=50,
)

print(test_metrics)
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
├── dcopf_gat/               # Core ML library
│   ├── data.py              # Data loading & preprocessing
│   ├── graph.py             # Graph construction & positional encodings
│   ├── model.py             # GAT + physics-informed loss
│   ├── train.py             # Training utilities
│   ├── runtime.py           # CPU/GPU runtime configuration
│   ├── data_pipeline.py     # tf.data input pipelines
│   └── utils.py             # Helpers (normalization, seeding)
│
├── scripts/
│   └── run_experiment.py    # Main training entrypoint
│
├── make_toy_3bus.py         # 3-bus synthetic dataset
├── make_toy_ieee14.py       # IEEE-14 synthetic dataset
├── experiments/             # Jupyter notebooks
├── data/                    # Placeholder for real datasets
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


