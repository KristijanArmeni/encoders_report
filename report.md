---
title: "Report: Encoding Model Reproduction"
bibliography: library.bib
authors:
  - name: Kristijan Armeni
    affiliation: jhu
  - name: Gabriel Kressin Palacios
    affiliation: jhu
  - name: Gio Li
    affiliation: jhu
---

<!-- *[Documentation](https://github.com/GabrielKP/enc/)* -->

# Introduction

We reproduced fitting the encoding model reported in @lebel_natural_2023.
We developed a repository to load data, compute features, and fit an encoding model to the dataset published by @lebel_natural_2023.
Encoding models are a popular method to make and test predictions about representational spaces in the brain.
The dataset contains pre-processed fMRI BOLD responses of eight participants that listened to 27 natural stories, their cortical surfaces, transcriptions for the stories.
It is accompanied by a repository to fit an encoding model[^lebel_code_repository], which we partly based our code on.
We fitted an encoding model with time-smoothed word vectors reproducing [Fig. 3B and 3E](https://www.nature.com/articles/s41597-023-02437-z/figures/3), and additionally fit an encoding model to the audio envelope.


[^lebel_code_repository]: https://github.com/HuthLab/deep-fMRI-dataset



# Methods

## Dataset

The dataset used for this report was described in @lebel_natural_2023 and is available in the OpenNeuro repository [@lebel_fmri_2023].

## Code

The code and its [documentation](https://gabrielkp.com/enc/) is available in a standalone GitHub repository[^github_repo].

[^github_repo]: https://github.com/GabrielKP/enc/

## Feature preprocessing

### Word embeddings

For each token in the story, its precomputed 985-dimensional embedding based on word co-ocurrences [@huth_natural_2016] were extracted from the [english1000sm.hfpy](https://github.com/OpenNeuroDatasets/ds003020/blob/main/derivative/english1000sm.hf5) data matrix, available in the OpenNeuro repository. If a story token was not available in the precomputed vocabulary, we filled that embedding with a zeros vector. The output of this step is a $N^{tokens} \times N^{dim}$ matrix of word embeddings.

**Aligning word timing with TR times.** To align the timings of words in each story with the sampled fMRI timecourses (TRs). We first constructed an array of word times $T^{word}$ by assigning each word a time half-way between it's starting time and offset time (l. 94 in https://github.com/GabrielKP/enc/blob/d34c32678647360339657225eeaea0e44801e4fc/src/main.py#L94). We then constructed an array of fMRI TR times $T^{fmri}$ (l. 95 in https://github.com/GabrielKP/enc/blob/d34c32678647360339657225eeaea0e44801e4fc/src/main.py#L95) by generating a array of TR indices spaced apart by the lenght of the TR (2 seconds) (e.g. resulting in $\mathrm{indices} = [0, 2, 4, 6]$ for 4 TR times). These onset times were then shifted forward in time (i.e. onset trimmed) by adding an offset of 10 seconds (e.g. $\mathrm{indices} + 10 = [10, 12, 14, 16]$). Finally every TR onset time was futher shifted forward in time to a mid-point between its onset start end time (e.g. $[11, 13, 15, 17]$).

**Resampling** To match the sampling frequency of word embeddings and fMRI data for regression, we resampled the stimulus matrix to match the sampling rate of the BOLD data (.5 Hz). Specifically, we transformed the discrete embedding vectors, which are defined only at word times, into a continuous-time representation. This representation is zero at all timepoints except for the middle of each word $T^{word}$. We then convolved this signal with a Lanczos kernel (with parameter $a=3$) to smooth the embeddings over time and mitigate high-frequency noise. Finally, we resampled the signal at the TR times of the fMRI data to create the embeddings matrix used for regression.

### Audio envelope

Audio envelope was computed by taking the absolute value of the hilbert transformed wavfile data.
For each story, the envelope was trimmed at the end by dropping the final 10 TR's as implemented in
[features.trim()](https://github.com/GabrielKP/enc/blob/d34c32678647360339657225eeaea0e44801e4fc/src/features.py#L19)
method. The trimmed envelope was then downsampled to match the number of TRs in the fMRI data. All the steps are implemented in the [load_envelope_data()](https://github.com/GabrielKP/enc/blob/d34c32678647360339657225eeaea0e44801e4fc/src/main.py#L24) method.


## fMRI preprocessing

We used the already preprocessed fMRI data as shared in by [@lebel_fmri_2023].

## Regression and cross-validation

We used the ridge regression model in the accompanying code [^lebel_code_repository] from [@lebel_fmri_2023].
We previously implemented the ridge regression model using [RidgeCV from scipy](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html), however the model fit significantly worse.
The ridge regression model mapped BOLD responses to the word embeddings.
Both the embeddings and the BOLD responses were z-scored prior to regression.
For each voxel the regression first optimized the $\alpha \in np.logspace(1, 3, 10)$, and computed the correlations between the predicted and the actual response using the best weight vector.

Cross-validation was performed at the story level to ensure the independence of training and test data.
Specifically, we randomly sampled a subset of stories to serve as the training set and held out one constant story as the test dataset in each fold.
The performance of the encoding model was quantified by calculating the Pearson correlation between the actual BOLD responses and the predicted BOLD responses for the test story.
To obtain a robust estimate of model performance, we repeated this random sampling process multiple times, varying the selection of training and test stories in each iteration.
The average performance across these repetitions provided a reliable measure of the encoding model’s performance, reducing the risk of performance being biased by any particular split of the data.

# Results

## Semantic encoding model

```{figure} fig/embeddings_performance.png
:label: fig-embedding
Test-set performance of the embeddings model.
```

## Audio envelope encoding model

```{figure} fig/envelope_performance.png
:label: fig-envelope
Test-set performance of the audio encoding model.
```

# Discussion



# List of challenges

* Alignment
* Parameters for plots in paper
