---
title: "Reproducing neuroscience analyses through collaborative open source tools and practices"
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

Computational tools are an indispenible part of contemporary science. Concurrently, computational reproducibility -- the ability for a different research team reproducing the original results given the original research artifacts -- has been recognized as a core standard for evaluating scientific claims [@claerbout_electronic_1992; @peng_reproducible_2009]. This holds as well for cognitive computational neuroscience which relies on compute-intensive methods for analyzing and modeling datasets of human brain responses during various perceptual and cognitive tasks [@naselaris_cognitive_2018]. As scientific insights increasingly depend on sophisticated and complex computational techniques, tools, and resources, researchers are encouraged to deliver robust and reproducible research artficats such as data, code, and documentation in addition to written reports [@donoho_invitation_2010; @peng_reproducible_2011; @sandve_ten_2013; @dupre_beyond_2022].

In computational neuroscience, machine learning methods are commonly used to make and test predictions about the kind of stimulus features (based on descriptions of stimuli or theoretical considerations) elicit and shape responses in the brain.
Statistical models that allow us to predict brain activity on the basis of stimuli presented to participants are known as encoding models [@naselaris_encoding_2011; @holdgraf_encoding_2017].[^decoding_models]
Encoding models can be used to answer basic scientific questions about the brain function [@kriegeskorte_cognitive_2018; @doerig_neuroconnectionist_2023] or to build clinical applications such as speech prostheses [@silva_speech_2024].

[^decoding_models]: Statistical mapping can also be performed in reverse direction by predicting classes or properties of stimuli on the basis of brain activity, in which case such statistical models are called 'decoding models'.

The prominence of machine learning and other computational approaches in neuroscience has resulted in a need for dedicated high-quality brain datasets that can be used for model training [e.g. @schoffelen_204-subject_2019; @armeni_10-hour_2022; @lebel_fmri_2023], data standards for sharing [@poldrack_making_2014; @gorgolewski_brain_2016], and targeted efforts towards identifying the barriers and solutions for reproducibility [@poldrack_computational_2019; @botvinik-nezer_variability_2020].

Here, we set out to reproduce the evaluation results for fMRI encoding models reported in a publicly-released datased @lebel_natural_2023. The goals of the project were two-fold:

1. To learn hwo to implement a novel analysis method by reproducing a set of published results in computational neuroscience.
2. To do so while adopting a set of software development practices and using a suite of open-source analysis and publishing tools.  

We developed a repository to load data, compute features, and fit an encoding model to the dataset published by @lebel_natural_2023. The dataset contains pre-processed fMRI BOLD responses of eight participants that listened to 27 natural stories, their cortical surfaces, and transcriptions for the stories.
It is accompanied by a repository to fit an encoding model[^lebel_code_repository], which we partly based our code on.
We fitted an encoding model with time-smoothed word vectors reproducing [Fig. 3B and 3E](https://www.nature.com/articles/s41597-023-02437-z/figures/3), and additionally fit an encoding model to the audio envelope.


[^lebel_code_repository]: https://github.com/HuthLab/deep-fMRI-dataset

# Methods

## Dataset

The dataset used for this report was described in @lebel_natural_2023 and is available in the OpenNeuro repository [@lebel_fmri_2023]. For computational and resource reasons, this reports focuses on a subset of results for one participant (`UTS02`). This specific participant dataset was chosen because the original report indicated it contained one of the best quality data based on the encoding model results reported by @lebel_fmri_2023. 

## Code

The code and its [documentation](https://gabrielkp.com/enc/) is available in a standalone GitHub repository[^github_repo]. Our approach can be dubbed `same-data-different-code` meaning that we used the original, published data and that we implemented our the same analysis using (in most parts) our own code, rather than used the original experiment code.

[^github_repo]: https://github.com/GabrielKP/enc/

## Feature preprocessing

(methods-embeddings-model)=
### Word embeddings

For each token in the story, its precomputed 985-dimensional embedding based on word co-ocurrences [@huth_natural_2016] were extracted from the [english1000sm.hfpy](https://github.com/OpenNeuroDatasets/ds003020/blob/main/derivative/english1000sm.hf5) data matrix, available in the OpenNeuro repository. If a story token was not available in the precomputed vocabulary, we filled that embedding with a zeros vector. The output of this step is a $N^{tokens} \times N^{dim}$ matrix of word embeddings.

**Aligning word timing with TR times.** To align the timings of words in each story with the sampled fMRI timecourses (TRs). We first constructed an array of word times $T^{word}$ by assigning each word a time half-way between it's starting time and offset time (l. 94 in https://github.com/GabrielKP/enc/blob/d34c32678647360339657225eeaea0e44801e4fc/src/main.py#L94). We then constructed an array of fMRI TR times $T^{fmri}$ (l. 95 in https://github.com/GabrielKP/enc/blob/d34c32678647360339657225eeaea0e44801e4fc/src/main.py#L95) by generating a array of TR indices spaced apart by the lenght of the TR (2 seconds) (e.g. resulting in $\mathrm{indices} = [0, 2, 4, 6]$ for 4 TR times). These onset times were then shifted forward in time (i.e. onset trimmed) by adding an offset of 10 seconds (e.g. $\mathrm{indices} + 10 = [10, 12, 14, 16]$). Finally every TR onset time was futher shifted forward in time to a mid-point between its onset start end time (e.g. $[11, 13, 15, 17]$).

**Resampling** To match the sampling frequency of word embeddings and fMRI data for regression, we resampled the stimulus matrix to match the sampling rate of the BOLD data (.5 Hz). Specifically, we transformed the discrete embedding vectors, which are defined only at word times, into a continuous-time representation. This representation is zero at all timepoints except for the middle of each word $T^{word}$. We then convolved this signal with a Lanczos kernel (with parameter $a=3$) to smooth the embeddings over time and mitigate high-frequency noise. Finally, we resampled the signal at the TR times of the fMRI data to create the embeddings matrix used for regression.

(methods-audio-model)=
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

Below, we report results for two fMRI encoding models: based on [distributional word embeddings](#methods-embeddings-model) ('semantic model') and based on [audio envelope](#methods-audio-model) ('sensory model').

## Semantic encoding model

In [](#fig-embedding) we show the test-set correlation results across the whole brain for participant `UTS02`. The highest performance is achieved with with the largest traininset (20 training stories). The best performing voxels are found in the bilateral temporal, parietal, and prefrontal cortices which is broadly in line with the spatial patterns in the original report [@lebel_fmri_2023]. The best performing voxels showed correlation values of ~0.35, which is lower than in the original report where highest scores reach a correlation of ~0.7. That is, our models pick up on the signal in the relevant brain areas, they are underporfming relative to original results.


```{figure} #cell.fig.embeddings
:label: fig-embedding

Test-set performance of the embeddings model with different training set sizes. Brigher color-coded voxels indicate better model performance. Test-set performance (Pearson correlation) is averaged across $N = 15$ independent models that were trained by resampling the training set 15 times.
```

## Sensory encoding model

To additionally our benchmark semantic model results, we implemented a simpler fMRI encoding model based on just the instantaneous envelope of the acoustic energy around word onsets. Our guiding expectation was that such a model would show spatially differnent patterns from from the more complex and statistically powerful semantic encoding model. [](#fig-envelope) displays the results. We see that compared to the semantic model ([](#fig-embedding)), the performance is has a much n arrower spatial extent and predominantly capures signal surrounding the primary auditory cortex (labeled AC) as expected by a low-level sensory model.

```{figure} #cell.fig.envelope
:label: fig-envelope
Test-set performance of the audio encoding model. The peak performance is observed in auditory cortex (AC).
```

## Model performance with increasing training set size

@lebel_fmri_2023 report that in general test-set performance increases with the increasing training set size (i.e. number of stories used to fit the model) increases. We sought to establish that the same trend holds for our pipeline and the two encoding models. We fit models for $N \in \{1, 3, 5, 7, 9, 11, 12, 15, 20\}$ stories. Results are shown in [](#fig-training-curve). 

The results confirm the increase in performance with more training data, however as noted above, our model undeperforms relative to published results exhibting lower correlation scores over all.

```{figure} fig/lebel_regression/training_curve.png
:width: 80%
:label: fig-training-curve
Test-set performance with increasing trainin set size (i.e. number of stories) for dataset UTS02.
```