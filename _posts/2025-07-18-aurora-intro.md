# Auora: Microsoft's Foundational Data-Driven Weather Model

Initially submitted to arXiv in May 2024, Aurora is Microsoft’s open-source foundation model for weather and climate prediction. Published in [Nature](https://www.nature.com/articles/s41586-025-09005-y) in May 2025, it’s now freely available on [GitHub](https://github.com/microsoft/aurora/tree/main), including both the code and the trained weights. Microsoft positions Aurora not just as another weather forecasting tool, but as a general foundation model for the atmosphere—capable of integrating diverse datasets and supporting a wide range of applications.

In this post, we’ll dive into how Aurora works under the hood and explore what makes it different from previous AI models for weather prediction.

## Datasets
A major point that differentiates Aurora from previous models is its various training datasets.  Most deep learning weather models in recent years have been trained primarily on ERA5, the widely used reanalysis dataset produced by ECMWF. However, for its pre-trained backbone, Aurora combines ERA5 with nine additional datasets, spanning different spatiotemporal resolutions, sets of variables, and numbers of pressure levels. It goes even further for its downstream fine-tuning, where depending on the task, the model is fine-tuned on an additional one or two datasets. This is a major break from previous models such as FourCastNet and GraphCast, that were only trained on a single dataset, ERA5.

This allows the model to learn from diverse sources of information, ranging from global climate reanalyses to higher-resolution simulations and observational datasets. Such heterogeneity is central to Aurora’s design, because it allows the model to remain flexible across tasks that differ in scale, variable coverage, or domain.

At the same time, heterogeneity poses a challenge for the model’s architecture. In other models like GraphCast or FourCastNet, the number of input variables does not change during training, whereas in Aurora’s case it might. This means the model needs to take this into account from an architectural perspective as well as a data infrastructure one.

## Architecture
Aurora’s architecture follows the classic encoder-processor-decoder pattern.  

Aurora represents all variables as 2D images on a latitude–longitude grid. For each variable, we take two snapshots: the current state (t) and the previous state (t−1). Stacking these gives a small time dimension (T = 2). The model treats atmospheric and surface variables separately. 
For atmospheric variables: if there are $V_A$ variables across $C$ pressure levels, the tensor looks like `V_A × C × T × H × W`.
For surface variables: if there are $V_S$ variables, the tensor looks like `V_S × T × H × W`.

Aurora also uses three "static" variables that never change with time:
- Surface geopotential (Z), which encodes topography
- Land–sea mask (LST)
- Soil-type mask (LSM)

Like in Vision Transformers, the H×W grids are split into P×P patches. Each patch is mapped to a vector of dimension D. 
- For atmospheric variables: `C × V_A × T × P × P → C × D`.
- For surface variables: `VS × T × P × P → 1 × D`.

Because datasets differ in what variables they provide, each variable $v$ has its own projection weights $W_v$, meaning that each variable type is passed through its own multi-layer perceptron with variable-specific weights.

To account for vertical structure, embeddings are tagged with level encodings. Pressure levels are represented either by a sinusoidal encoding of the pressure value (e.g. 150 hPa) for atmospheric data or by learned vectors of size $D$ for surface data.

Since different datasets have different numbers of atmospheric pressure levels, these embeddings are then aggregated through a [Perceiver module](https://arxiv.org/abs/2103.03206), in which latent query vectors attend to the encoded pressure levels. In particular, the latent query vector has a size of $C_L=3$ such that the latent representation is $C_L \times D$.  This essentially maps a variable number of pressure levels, `C` (depending on the dataset), to a fixed set, $C_L$. 

In parallel, the surface embedding goes through a residual MLP. The outputs are concatenated to form a $(C_L + 1) \times D$ representation of the full weather state at each patch.

<figure>
  <img src="/assets/graph1.png" alt="Graph" width="300" height="300" class="center-image">
  <figcaption class="figcaption-2">Fig. 1: Graph with 7 nodes and 6 edges</figcaption>
</figure>

