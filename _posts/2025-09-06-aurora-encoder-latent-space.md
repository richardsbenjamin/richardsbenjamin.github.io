# Exploring the Latent Space of Aurora's Encoder
In a previous [post](https://richardsbenjamin.github.io/2025/08/30/aurora-intro.html), I introduced Microsoft’s Aurora model, outlining its architecture and the motivations behind its design.

In this post, we’ll take a closer look inside the latent space of the model’s encoder to examine what kinds of representations it has learned. The role of the encoder is to compress inputs into an embedded representation, while the processor component is responsible for modeling temporal dynamics. Because of this division of labor, we should not expect the encoder to capture explicit information about physical processes or time evolution. Instead, we might uncover structural distinctions present in the raw input data itself.

Specifically, this analysis will focus on whether the latent space encodes a clear separation between land and ocean. If such a distinction emerges, it would suggest that the encoder is capturing meaningful features tied to the geography of the input fields, an encouraging sign that the model is developing representations aligned with real-world structure. If not, it may point to limitations in the encoder’s ability to disentangle different components of the input space.

## What is the Latent Space?
The latent space is a compressed, numerical representation (often a high-dimensional vector) produced by the encoder from raw inputs such as global temperature fields, wind patterns, or precipitation maps. It acts as a bottleneck layer where the model distills the most important spatial and statistical features of the atmosphere and surface conditions into a compact form.

Exploring this space helps reveal how the model internally organises weather information. By probing the latent space, we can better understand not just what the model predicts, but how it represents the underlying weather features it relies on.

## The Encoder Ouput
The encoder's output is a matrix of size $512\times 259,200$. To interpret it, it will be fruitful to understand how it was constructed. 
 
Surface and static inputs are combined into a tensor of size $2\times 7 \times720\times 1440$ (two time steps, seven variables, global grid). With a patch size of 4 and embedding dimension of 512, the surface encoder produces $512\times 180\times 360$, which is then flattened to $512\times 64800\times 1$.

Atmospheric inputs start as $2\times 5\times 13\times 720\times 1440$ (two time steps, five variables, 13 pressure levels). Using the same patching scheme, this becomes $512\times 64800\times 3$. 

Stacking surface and atmospheric embeddings yields $512\times 64800 \times4$, which is flattened again to the final $512\times 259200$. Each column is an embedded vector representing either a surface patch or an atmospheric patch at a given level. Together, they cover the full globe.

This structure is important because later we'll want to map vectors back to their original patches—for example, to test whether the latent space separates land from ocean or different atmospheric regimes.

To generate these embeddings, we pass the input fields directly through Aurora’s encoder. 

```python

aurora_model = get_aurora_model(device)

aurora_static = AuroraStatic(drive_path)
batch = get_aurora_data(ic, aurora_static)

# Before passing to the encoder, we need to process the input data
# according to their code
p = next(aurora_model.parameters())
transformed_batch = batch.type(p.dtype)
transformed_batch = transformed_batch.normalise(surf_stats=aurora_model.surf_stats)
transformed_batch = transformed_batch.crop(patch_size=aurora_model.patch_size)
transformed_batch = transformed_batch.to(p.device)

B, T = next(iter(transformed_batch.surf_vars.values())).shape[:2]
transformed_batch = dataclasses.replace(
    transformed_batch,
    static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in transformed_batch.static_vars.items()},
)

# Get the embedding, shape (1, 259200, 512)
full_embedding = aurora_model.encoder(transformed_batch, aurora_model.timestep)

# Deconstruct into surface and atmospheric embeddings
reshaped_embedding = res.reshape(1, 4, 64800, 512).squeeze()
surf_embedding = reshaped_res[0].transpose(1, 0)
atmos_embedding = reshaped_res[1:]

```

## Ocean and Land

Next, we test whether the encoder has learned to distinguish ocean from land. During training, Aurora receives a land–sea mask as input, so it's reasonable to expect this distinction to appear in the latent space. Importantly, this separation matters physically as land and ocean respond differently to forcing, so learning this boundary is consistent with the underlying dynamics. However, there is no guarantee that it has learnt this boundary. 

Our analysis uses two approaches: principal component analysis (PCA) for visualisation, and logistic regression with the land–sea mask as labels to quantify separability.

To prepare the labels, we start from Aurora’s static land–sea mask and downsample it by the patch size. A patch is classified as land if more than 50% of its area is land, otherwise it’s ocean. This gives us patch-level labels aligned with the encoder's embeddings.

```python
import xarray as xr

static = xr.open_dataset("static.nc")

patch_size = 4

land_sea_mask = static["lsm"].values[:, :720, :,].squeeze()
land_sea_mask_patched = reduce_mask(land_sea_mask, patch_size)

```

### PCA
PCA is a machine learning technique that identifies the directions of maximum variance in a dataset. In our context, it helps reveal the dominant modes of variation in Aurora's latent space—showing whether the model organises patches by features such as land–ocean boundaries, latitude bands, or large-scale climate gradients.

By projecting the high-dimensional embeddings onto the first few principal components, we can visualise and interpret how the encoder structures weather and climate information.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
all_surface_vectors_2d = pca.fit_transform(surf_embedding.T)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(all_surface_vectors_2d[:, 0], all_surface_vectors_2d[:, 1],
                     c=land_sea_mask_patched.ravel(), cmap='viridis', alpha=0.6, s=2)
plt.colorbar(scatter, label='Is Land? (0=Ocean, 1=Land)')
plt.xlabel('Principal Component 1 ({:.2f}% Var)'.format(pca.explained_variance_ratio_[0]*100))
plt.ylabel('Principal Component 2 ({:.2f}% Var)'.format(pca.explained_variance_ratio_[1]*100))
plt.title('PCA of Surface Latent Vectors: Land vs. Ocean')
plt.show()
```

The output is shown below. 

<figure>
  <img src="/assets/aurora_encoder_pca.png" alt="Graph" width="300" height="300" class="center-image">
  <figcaption class="figcaption-2">Fig. 1: PCA visualuation of surface embedding</figcaption>
</figure>

The separation between land and ocean is evident in the PCA space, with yellow points representing land and purple points representing ocean. Although these points partially overlap, they also tend to organise into distinct spatial regions. Oceans are primarily located in the left and central areas of the plot, corresponding to low to moderate values of PC1, while land points are concentrated toward high positive values of PC1, showing a wide dispersion along PC2.

This indicates a clear but not perfect separation, as the two clusters are not entirely disjoint. The visible trajectories suggest an underlying temporal or spatial organization, and the curved shapes of the points imply that the latent variables are constrained by different physical regimes over land and sea. 

The overlap indicates that some oceanic regions share characteristics with land, such as coastal zones or enclosed seas, and vice versa. Overall, the PCA reveals a marked separation between the latent vectors of land and ocean, with high explained variance, showing that the latent space effectively encodes this physical distinction, even though intermediate zones exist.

### Logistic Regression
We next apply logistic regression to predict whether a patch corresponds to land or ocean, using the latent vectors as input. This tests directly whether the encoder has encoded the land–sea boundary.

For evaluation, we split the globe into training and testing regions. All patches between longitudes 120° and 210° (covering much of Australia and East Asia) form the test set, giving roughly a 75/25 split.

After classification, we can also map the errors back onto the globe to see where the regression fails, highlighting regions where the encoder's representation of land–sea differences is less distinct.

```python
test_lon_min = 120.0
test_lon_max = 210.0

train_split_dict = get_train_test_split(
    test_lon_min,
    test_lon_max,
    patch_center_lon,
    land_sea_mask_patched.ravel(),
    surf_embedding,
)

reg_res = run_logistic_regression(train_split_dict)

```

Running the regression gives an accuracy of **99.87%**, a clear indication that the encoder has internalised the land–sea distinction. Still, this result comes from a single run and a relatively simple task, so it should be interpreted cautiously.

To dig deeper, we can examine the misclassified patches to see where the model struggles to maintain this separation.

```python
# Get the locations of the errors
is_misclassified = (reg_res["y_pred"] != train_split_dict["y_test"])

error_lons = region_center_lons[is_misclassified]
error_lats = region_center_lats[is_misclassified]

# Plot them
plot_dots_on_map(
    error_lons, error_lats,
    color="black", s=50, alpha=0.7,
    extent=[-150, 90, -90, 90],
)
```

<figure>
  <img src="/assets/aurora_log_reg_errors.png" alt="Graph" width="300" height="300" class="center-image">
  <figcaption class="figcaption-2">Fig. 2: Visualising where the locations of the logistic regression errors.</figcaption>
</figure>

Most errors occur along coastlines, where the land–sea distinction is inherently less clear. This is an encouraging result: the encoder not only separates land from ocean but also reflects the natural uncertainty present at boundaries.













