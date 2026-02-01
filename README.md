# Hybrid MAML-STGCN-LSTM for Global-to-Regional Weather Forecasting (Model v5.0)

## Overview

This repository contains **Model v5.0**, a meta-learning–based spatio-temporal graph neural network for **weather forecasting**, designed to **generalize across global climate regions and rapidly adapt to new, unseen locations** using limited local data.

The model integrates:
- Spatio-Temporal Graph Convolutional Networks (STGCN) for spatial dependency modeling
- Deep multi-layer LSTM for long-range temporal forecasting
- Model-Agnostic Meta-Learning (MAML) to enable fast regional adaptation

The system is trained on **globally distributed climate regions** and fine-tuned to **specific regions** (e.g., deserts, tropics, polar zones) using multi-year historical data.

---

## Model Architecture

### Core Components

- **STGCN Backbone**
  - Captures spatial dependencies between geographic grid points
  - Graph constructed using k-nearest-neighbor connectivity
- **LSTM**
  - Deep LSTM stack for temporal sequence modeling
  - Improves long-horizon forecast stability
- **MAML Meta-Learning Framework**
  - Learns initialization parameters that adapt quickly to new regions
- **Köppen Climate Embedding**
  - Climate-type–aware embeddings injected into model inputs

### Model Scale

| Component | Value |
|--------|------|
| STGCN Hidden Channels | 256 |
| LSTM Hidden Size | 128 |
| LSTM Layers | 4 |
| Input Channels | 24 |
| Output Channels | 12 |
| Input Window | 24 time steps |
| Forecast Horizon | 8 time steps |

---

## Dataset

### ERA5 Reanalysis (Single Levels)

This model is trained using the **ERA5 hourly single-levels reanalysis dataset** from **cds.climate.copernicus.eu**

**Dataset characteristics**
- Temporal resolution: Hourly
- Spatial resolution: 0.25° × 0.25° (30 km)
- Coverage: Global
- Data type: Surface and single-level atmospheric variables
- Format: NetCDF

ERA5 provides physically consistent and globally complete weather fields, making it suitable for spatio-temporal learning and climate generalization tasks.

---

## Training Data Details

### Meta-Training (Global Training)

- Number of regions: 15 globally distributed regions
- Climate coverage:
  - Tropical, temperate, desert, polar, and monsoon climates
- Sampling strategy:
  - Each region is treated as an independent meta-learning task
- Per-region data usage:
  - Up to ~600 spatio-temporal samples per region
  - Support / query split:
    - ~75% support (inner-loop adaptation)
    - ~25% query (meta-update)


Each training sample consists of:
- A 24-step spatio-temporal input window
- An 8-step forecast horizon
- A graph-structured spatial representation

---

### Regional Adaptation Data

After meta-training, the model is adapted to new regions using **multi-year historical data**.

- Years used: 2023–2024
- Seasonal coverage: All four quarters per year
- Sampling cap: Up to ~1,200 windowed samples per region
- Train / validation split:
  - 80% training
  - 20% validation


Each training sample consists of:
- A 24-step spatio-temporal input window
- An 8-step forecast horizon

---

### Regional Adaptation Data

After meta-training, the model is adapted to new regions using **multi-year historical data**.

- Years used: 2023–2024
- Seasonal coverage: All four quarters per year
- Sampling cap: Up to ~1,200 windowed samples per region
- Train / validation split:
  - 80% training
  - 20% validation

## Training Strategy

### Meta-Learning (MAML)

- **Inner Loop**
  - SGD-based fast adaptation
  - Multiple inner epochs per task
- **Outer Loop**
  - AdamW optimizer
  - Gradient accumulation
  - Cosine annealing with warm restarts
- **Stability techniques**
  - Gradient clipping
  - Adaptive task sampling based on task difficulty

---

## Regional Adaptation

The pretrained model has been adapted and evaluated on diverse regions, including:
```
MODEL_REGIONS = [
    (18, 23, 75, 80),  # India
    (8, 13, 98, 103),  # Thailand
    (53, 58, 35, 40),  # Russia
    (12.5, 17.5, 102.5, 107.5),  # Thailand/Cambodia
    (22.5, 27.5, 19.5, 24.5),  # Libya/Egypt
    (43.5, 48.5, 7.5, 12.5),  # Southern France
    (35.5, 40.5, -5.5, -0.5),  # Spain/Mediterranean
    (32.5, 37.5, 137.5, 142.5),  # Tokyo/Eastern Japan
    (-23.5, -18.5, 132.5, 137.5),  # Australia
    (-20, -15, -70, -65),  # Peru
    (44.5, 49.5, 125.5, 130.5),  # Northeast China
    (29.5, 34.5, -101.5, -96.5),  # Texas
    (-9.5, -4.5, -67.5, -62.5),  # Amazon Basin
    (67.5, 72.5, -32.5, -27.5),  # Greenland
    (51.5, 56.5, -112.5, -107.5),  # Alberta, Canada
]
```

### Adaptation characteristics:
- Multi-year seasonal data
- Climate-aware optimizer and learning-rate scheduler
- Full fine-tuning of STGCN and LSTM layers
- Optimized for fast convergence

---

## Evaluation

 - Loss function: Mean Squared Error (MSE)

 - Validation performed on held-out regional datasets

---

## Limitations

 - Computationally intensive due to GNN + LSTM + MAML architecture

 - Requires gridded weather data in NetCDF format

 - Uses static spatial graphs

 - Not optimized for real-time inference

## Research Motivation

This project was developed as an academic research-oriented system focusing on:

 - Meta-learning for spatio-temporal forecasting

 - Climate-aware adaptation mechanisms

 - Graph-based representations of physical systems

The design emphasizes extensibility and reproducibility, making it suitable for research papers, final-year projects, and graduate-level coursework.

# Author

Yashas Gowda N

AIML Undergraduate

**Research interests**: Meta-Learning, Graph Neural Networks, Climate AI

## License

MIT License
