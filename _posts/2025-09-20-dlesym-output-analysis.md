# DLESyM Output Analysis

Several weeks ago, a new climate AI model [came out](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025AV001706), DLESyM.

In this post, we will go over the model's architecture, and some initial analysis regarding the model's stability and simulation of extreme events. 

## DLESyM Architecture

...VARS AND COUPLED MODEL
...ARCHI

In DLESyM, the ocean evolves more slowly than the atmosphere, so the two models run on different time steps. The ocean module (DLOM) updates every 4 days, producing sea-surface temperature forecasts at 48-hr and 96-hr intervals. The atmospheric module updates every 12 hours, outputting forecasts at 6-hr and 12-hr intervals.

To couple them, the system advances in 96-hr cycles. The atmosphere is first simulated forward in 12-hr steps, generating predictions up to 96 hr. These forecasts provide averaged fields, e.g. wind speed and surface pressure, which are then used to drive the ocean model forward over the same 96-hr window.

This process repeats iteratively, keeping the coupled system stable over long simulations. A key design choice is that near-surface air temperature is not passed to the ocean model, to avoid spurious feedbacks, since sea-surface temperatures primarily force the atmosphere, not the other way around.

## Output Analysis
The first thing to do was to generate the output. 

Since the output is roughly 250GB, I opted for storing the output in a Google bucket. To do this, I forked the origin repo and modified it so that the output data is saved to a zarr file. My fork can be found [here](https://github.com/richardsbenjamin/DLESyM). I also hired some GPUs to do the job, it took about an hour and a half. 

### Climate Drift
The first step in analysing the climate output is to calculate global averages over the whole simulation period for a given variable. In addition, we can calculate a linear regression line.

A flat line is ideal. While we expect natural variability, we also expect the regression line to be almost completely flat. A steep line, indicating a cooling or warming trend means the model is not stable. 

<figure>
  <img src="/assets/asd.png" alt="Graph" width="300" height="300" class="center-image">
  <figcaption class="figcaption-2">Fig. 1: ...</figcaption>
</figure>







