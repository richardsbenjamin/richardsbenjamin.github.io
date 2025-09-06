# Exploring the Latent Space of Aurora's Encoder
In a previous [post](https://richardsbenjamin.github.io/2025/08/30/aurora-intro.html), I gave an overview on Microsoft's Aurora model.

In this post, we are going peak inside the latent space of the model's encoder to see what kind of representations it has learnt. The intention is that the encoder learns an embedded representation of the inputs, whereas the model's processor learns the dynamics, and so we are unlikely to find representations of physical dynamics. However, we may be able to find certains distinctions present in the input space. 

In this analysis, we'll be looking at whether the latent space has learnt the difference between ocean and land, and...

## What is the Latent Space?
The latent space is a compressed, numerical representation (often a high-dimensional vector) that the model's encoder produces from an input (e.g., an image, audio clip, or molecule). It's a "bottleneck" layer where the model encodes the most salient features of the input in a way it finds useful for its task (e.g., reconstruction, prediction, generation).

The goal of your exploration is to understand the structure and semantics of this space. What directions correspond to meaningful features? How are different concepts organized?


## The Encoder Ouput
If you read Aurora's paper (and my last post), you'll understand that the encoder's output is a...

The output of the Aurora's encoder is a matrix of size $512x259200$. To do some analysis on the space, it will be fruitful to understand how it was constructed. 

Firstly, the surface, static and atmospheric data are passed separately as input to the encoder. 

The surface and static data are concatenated together to give a tensor of size $2x7x720x1440$. Here we have 2 time steps, 7 variables in total, 720 longitudes points and 1440 latitude points. The patch size of the surface encoder is 4, and the embedded dimension is 512. This produces an embedded output of $512x180x360$. They flatten the last two dimensions and arrange it as $512x64800x1$.

For the atmospheric data, we start with a tensor of size $2x5x13x720x1440$; 2 time steps, 5 variables, 13 atmospheric levels, and the longitude and latitude points. We have the same patch size and embedded dimension as the surface encoder, but we also have the atmospheric variables to be compressed together. Because of this, the resulting emebedded output is $512x64800x3$. 

For the next step, we stack the surface and atmospheric embeddings together to obtain a tensor of size $512x64800x4$, and we flatten the last two dimensions again to produce the final output of $512x259200$. With this, we have 259,200 embedded vectors, and because of this structure, for a given embedded vector, it will either be a representation of the surface or one of the atmospheric levels. However, note that all of the embedded surface vectors together represent the either plant (the full longitude / latitude range), same for the atmospheric levels, and a single vector represents a patch. 

This will be useful to know because later on, we will want to map embedded vectors back to the original patch to execute some classification tasks on the embedded space. 

To get the Aurora ouput, we need to load the model and pass the input directly to the encoder. 

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

We are going to explore whether the encoder has learnt the difference between ocean and land. When training the aurora model, we pass an ocean and land mask, so it makes sense that it has learnt the difference between the two. However, it's only learnt this difference because it has been deemed relevant to the predictions, which is true the in actualy dynamic system, so this is evidecne that the encoder has learnt real physics. 

For the analysis, we will perform principal component analysis (PCA) and visualise the results. We'll also perform logistic regression using the land-sea mask as labels.

One of the first things that we need to do is generate the land-sea labels. We start the land-sea mask from the Aurora static, but then we need to reduce the mask by a factor of the patch size. This way we obtain the land sea mask for the patches (the patch is consdired land if more than 50% of the patch is land). 

```python
import xarray as xr

static = xr.open_dataset("static.nc")

patch_size = 4

land_sea_mask = static["lsm"].values[:, :720, :,].squeeze()
land_sea_mask_patched = reduce_mask(land_sea_mask, patch_size)

```

### PCA
PCA is a machine learning technique that finds the directions of maximum variance. It's crucial for understanding the global primary axes of variation in your data. 

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
Next we'll be doing a logistic regression, attempting to predict whether a given patch is ocean or land using the patch embedding as input. This allows us to assess whether the latent vectors have encoded the land-sea distinction. 

We'll split the vector space into a train and test region. The test region will be all point between longitude 120 and 210. This encompasses much Australia and east Asia, and corresponds roughly to a 75/25 train/test split. 

After having done the classification, we'll also be able to visualise the points on the map where the regression was wrong. 

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

From when I ran it, the accuracy was **99.87%**. An extremely positive result, indicating that encoder has learnt the land-sea distinction. However, don't forget that this is only one sample, and remains a fairly simple classification task. 

We can further explore the results of the regression by inspecting its errors. 

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

Here we can see that most of the errors correspond to coastlines, whether the distinction between sea and land is less sharp. The fact that the errors correspond to coastlines is actually reassuring as even a certain level of uncertainty has been encoded.











