---
title: "Executable science: Research Software Engineering Practices for Replicating Neuroscience Findings"
short_title: Executable science
abstract: |
  Computational approaches are central to deriving insights in modern science. In this interdisciplinary convergence, researchers are increasingly expected to share research code and data along with published results. However, with software engineering practices still predominantly an overlooked component in research, it remains unclear whether sharing data and code alone is sufficient for reproducibility in practice. We set out to reproduce and replicate evaluation results of a publicly available human neuroimaging dataset recorded when participants listened to short stories. Using the code and data shared with the paper, we predicted the brain activity of the participants with semantic embeddings of words in stories. We successfully reproduced the original results, showing that model performance improved with more training data. However, when we replicated the analysis with our own code, the model's performance was much lower than reported in the paper. We traced this difference to the discrepancy between method description and its implementation in shared code. While access to original code proved crucial for our ability to replicate the results, barriers such as a complex code structure and insufficient documentation hindered our progress. We argue that software engineering practices are essential for reproducible and reusable research. We discuss components of our workflow and specific practices that effectively mitigate barriers to reproducibility.
keywords:
  - replication
  - reproducibility
  - fMRI
  - computational neuroscience
  - research software engineering
bibliography: library.bib
---

<!-- *[Documentation](https://github.com/GabrielKP/enc/)* -->

# Introduction

Computational methods are an indispensable part of contemporary science. For example, cognitive computational neuroscience is a nascent discipline using computational models to investigate the brain's representations, algorithms, and mechanisms underlying human intelligent behavior in healthy and clinical populations [@kriegeskorte_cognitive_2018; @naselaris_cognitive_2018]. As computational techniques become central to scientific discovery, the ability of independent research teams to reproduce and replicate published work has been recognized as a standard for evaluating the reliability of scientific findings [@claerbout_electronic_1992; @donoho_invitation_2010; @peng_reproducible_2009]. Researchers are now strongly encouraged to share open and reproducible research data, code, and documentation alongside written reports [@peng_reproducible_2011; @pineau_improving_2021; @jwa_spectrum_2022; @nelson_ensuring_2022], a trend also reflected in cognitive computational neuroscience [@sejnowski_putting_2014; @poldrack_making_2014; @gorgolewski_brain_2016; @poldrack_scanning_2017; @botvinik-nezer_reproducibility_2023; @de_vries_sharing_2023]. Despite this progress, achieving computational reproducibility in research remains a significant practical challenge, due to the increasing complexity of both data and analytical methods, as well as ethical and legal constraints [@open_science_collaboration_estimating_2015; @botvinik-nezer_variability_2020; @jwa_spectrum_2022; @hardwicke_analytic_2021; @xiong_state_2023; @barba_path_2024].

While there is growing momentum toward sharing research artifacts such as data and code, making them available does not guarantee reproducibility [@mckiernan_policy_2023]. To be truly reproducible, research artifacts must also be reusable, readable, and resilient [@connolly_software_2023].
Reproduction efforts can falter when artifacts are incomplete or contain errors -- such as missing data or faulty scripts [@pimentel_large-scale_2019; @hardwicke_analytic_2021; @nguyen_are_2025] -- or when the documentation is sparse or unclear, making it difficult to rerun analyses with the provided data [@xiong_state_2023].
In contrast, the software engineering community has long developed best practices for creating robust and reusable general-purpose artifacts [e.g., @thomas_pragmatic_2020; @beck_manifesto_2001; @martin_clean_2012; @washizaki_guide_2024].
However, these practices are not always directly applicable to research settings, where the scope of engineering needs varies from standalone analysis scripts to domain-specific research software tools [@balaban_ten_2021; @connolly_software_2023]. Given that much of scientific research is conducted within relatively small, independent teams, a central question arises: what software practices and open-source tools are best suited to support reproducible research?

Here, we share our experience and lessons learned from a small-scale reproducibility project in an academic neuroscience lab. We set out to reproduce and replicate the evaluation of predictive models on a publicly available fMRI dataset, which demonstrated that increasing the amount of training data improved average performance in predicting brain activity at the individual participant level [@lebel_natural_2023]. We conducted three complementary experiments: i) a **reproduction** experiment using the original shared data and code, ii) a **replication** experiment using the shared data but re-implementing the analysis with our own code, and iii) an **extension** experiment using the shared data in a novel analysis not reported in the original paper.

The main contributions of our work are:

- We successfully reproduced the original results using the shared data and code.
- We identified several challenges when replicating the analysis with our own implementation.
- We describe components of our reproducible workflow and review software engineering practices that can facilitate reproducible research.

# Methods

In what follows, we provide a high-level overview of the methodology, the data used, and the analysis design for our experiments following the original experiment and dataset [@lebel_natural_2023].

## Encoding models

A central goal in computational neuroscience is to test hypotheses about the computations and representations the brain uses to process information. Encoding models address this goal by quantifying the relationship between features of a stimulus and the corresponding neural responses to the same stimulus.

In practice, machine learning methods are often used to extract relevant stimulus features (e.g., semantic, visual, or acoustic), which are then used in statistical models (typically linear regression [@ivanova_beyond_2022]) to predict neural activity [@naselaris_encoding_2011]. By comparing the predictive performance of different encoding models (e.g., from different stimulus features across different brain areas), researchers can draw inferences about the spatial and temporal organization of brain function [@kriegeskorte_interpreting_2019; doerig_neuroconnectionist_2023]. Encoding models are also foundational for brain-computer interfaces, including clinical applications such as speech prostheses, which aim to reconstruct intended speech from recorded brain activity [@silva_speech_2024].

Formally, an encoding model estimates a function that maps stimulus features $\textbf{X}$ to brain responses $\textbf{Y}$. The model typically takes a linear form:

```{math}
\hat{\textbf{Y}} = \textbf{X} \hat{\textbf{W}}
```

where {math}`\hat{\textbf{Y}} \in \mathbb{R}^{T \times N}` denotes predicted brain responses across $T$ time points and $N$ voxels, and {math}`\textbf{X} \in \mathbb{R}^{T \times D}` contains $D$-dimensional stimulus features. The weight matrix {math}`\hat{\textbf{W}} \in \mathbb{R}^{D \times N}` captures the contribution of each feature dimension to each voxel’s activity.

Model estimation involves fitting $\hat{\textbf{W}}$ to predict brain activity $\hat{\textbf{Y}}_{\text{train}}$ from features $\textbf{X}_{\text{train}}$ using statistical methods such as ridge regression. Model performance is then evaluated on held-out test data by comparing predicted brain activity $\hat{\textbf{Y}}_{\textsf{test}}$ to observed brain activity $\textbf{Y}_{\textsf{test}}$ using a scoring function such as Pearson correlation: $r = \textsf{correlation}(\hat{\textbf{Y}}_{\textsf{test}}, \textbf{Y}_{\textsf{test}})$.

:::{note} Neuroscience Glossary
:class: dropdown

:::{glossary}
fMRI
: Functional magnetic resonance imaging (fMRI) is a non-invasive neuroimaging technique that measures brain activity by detecting changes of oxygenated blood flow. It is commonly used to study brain function by capturing dynamic activity signals across the whole brain.

BOLD  
: Blood-oxygen-level-dependent or BOLD is the signal measured in fMRI that reflects changes in the concentration of oxygenated versus deoxygenated blood. Neural activity leads to localized changes in blood flow and oxygenation, which can be detected as BOLD signal fluctuations.

TR
: Time-to-repeat (TR) is the time interval between successive measurements of the same slice or volume in fMRI. It determines the temporal resolution of the acquired data, typically ranging from 0.5 to 3 seconds.

Voxel
: The smallest unit of measurement in fMRI data, representing a fixed region of brain tissue in three-dimensional space. Each voxel contains a summary signal that reflects the average activity within that region, typically encompassing thousands of neurons.

:::

## Dataset

The publicly available fMRI dataset provided by @lebel_natural_2023 recorded blood-oxygen-level-dependent (BOLD) signals while participants listened to 27 natural narrative stories.
The dataset is available via the OpenNeuro repository[^openneuro].

[^openneuro]: https://openneuro.org/

### Preprocessed fMRI responses ({math}`\textbf{Y}_{\textsf{fmri}}`)

Because the focus of our work was on encoding models, we chose to use the preprocessed fMRI data provided by @lebel_natural_2023.
As a result, our reproducibility efforts did not involve preprocessing of raw fMRI data, which includes steps such as motion correction, cross-run alignment, and temporal filtering.
We limited our analysis to a subset of three participants ({math}`\textsf{S01, S02, S03}`) because they showed the highest signal quality and highest final performance in the original paper.
We reasoned that this minimal subset of data would be sufficient to evaluate the reproducibility and replicability of the modeling results.

We accessed the data using the DataLad data management tool [@halchenko_datalad_2021], as recommended by @lebel_natural_2023.
Time series were available as {file}`.h5p` files for each participant and each story. Following the original report, the first 10 seconds (5 TRs) of each story were trimmed to remove the 10-second silence period before the story began.

**Hemodynamic response estimation**  
The fMRI BOLD responses are thought to represent temporally delayed (on the scale of seconds) and slowly fluctuating components of the underlying local neural activity [@logothetis_underpinnings_2003]. To account for this delayed response, predictor features for timepoint {math}`t` were constructed by concatenating stimulus features from time points {math}`t - 1` to {math}`t - 4` ({math}`n_{\text{delay}}=4`). For timepoints where {math}`t-k<0`, zero vectors were used as padding. This resulted in a predictor matrix {math}`X \in \mathbb{R}^{T\times 4D}`. Although this increases computational cost, it enables the regression model to capture the shape of the hemodynamic response function [@boynton_linear_1996] underlying the BOLD signal.


## Predictors

### Semantic predictor ({math}`\textbf{X}_{\textsf{semantic}}`)

To model brain activity related to aspects of linguistic meaning understanding during story listening, @lebel_natural_2023 used word embeddings --- high-dimensional vectors capturing distributional semantic properties of words based on their co-occurrences in large collections of text [@clark_vector_2015].

**Extracting word embeddings and timings**  
For each word in the story, we extracted its precomputed 985-dimensional embedding vector [@huth_natural_2016] from the {math}`\textsf{english1000sm.hf5}`[^english1000sm] data matrix (a lookup table) provided by @lebel_natural_2023 and available in the OpenNeuro repository. Words not present in the vocabulary were assigned a zero vector. For every story, this yielded a {math}`\hat{\textbf{X}}_{\textsf{semantic}} \in \mathbb{R}^{N_{\text{words}} \times 985}` matrix of word embeddings for each story where {math}`N_{\text{words}}` are all individual words in a story. The onset and offset times (in seconds) of words were extracted from {math}`\textsf{*.TextGrid}` annotation files[^textgrid] provided with the original dataset.

[^english1000sm]: https://github.com/OpenNeuroDatasets/ds003020/blob/main/derivative/english1000sm.hf5

[^textgrid]: These are structured .txt files that are used with the Praat software for acoustic analysis (https://www.fon.hum.uva.nl/praat/)

**Aligning word embeddings with fMRI signal**  
The fMRI BOLD signal was sampled at regular intervals (repetition time, or TR = 2s). To compute a stimulus matrix {math}`\hat{\textbf{X}}_{\textsf{semantic}}` that matched the sampling rate of the BOLD data, following @lebel_natural_2023, we first constructed an array of word times {math}`T^{\text{word}}` by assigning each word a time half-way between its onset and offset time. This was used to transform the embedding matrix (which was at discrete word times) into a continuous-time representation. This representation is zero at all timepoints except for the middle of each word {math}`T^{\text{word}}`, where it is equal to the embedding vector of the word. We then convolved this signal with a Lanczos kernel (with parameter {math}`a=3` and {math}`f_{\text{cutoff}}=0.25` Hz) to smooth the embeddings over time and mitigate high-frequency noise. Finally, we resampled the signal half-way between the TR times of the fMRI data to create the feature matrix used for regression, {math}`\textbf{X}_{\textsf{semantic}} \in \mathbb{R}^{N_{\text{TRs}} \times N^{\text{dim}}}`.

### Sensory predictor ({math}`\textbf{X}_{\textsf{sensory}}`)

To benchmark the predictive performance of our semantic predictor, we decided to develop an additional, simpler encoding model based on acoustic properties of the stories (not reported in the original work by @lebel_natural_2023).
The audio envelope was computed by taking the absolute value of the hilbert-transformed wavfile data.
For each story, the envelope was trimmed at the end by dropping the 10 beginning and final seconds.
The trimmed envelope was then downsampled to the sampling frequency of the fMRI data.

## Ridge regression

To fit a penalized ridge regression model, we used the scikit-learn library [@pedregosa_scikit-learn:_2011], which is a mature library for machine learning with a large user-base and community support.
Specifically, we used the `RidgeCV()` class which performs a leave-one-out cross-validation to select the best value of the {math}`\alpha` hyperparameter for each target variable (i.e. brain activity time courses in each voxel) prior to fitting the model.
We set the possible hyperparamater values to {math}`\alpha \in` `np.logspace(1, 3, 10)` as in the original report [@lebel_natural_2023].
The {math}`\alpha` value that resulted in the highest product-moment correlation coefficient then was used by `RidgeCV()` as the hyperparameter value to fit the model.

## Evaluation

### Cross-validation

Cross-validation was performed at the story level to ensure the independence of training and test data.
Specifically, to construct the training set, we randomly sampled without replacement a subset of {math}`N^{\text{train size}}` stories from the {math}`N^{\text{total stories}} = 26` training set pool and held out one constant story (*Where there's smoke*) as the test dataset in each fold.
This random sampling process was repeated {math}`N^{\text{repeat}} = 15` times, with a new selection of training stories from {math}`N^{\text{total stories}}` in each iteration as described in the original report [@lebel_natural_2023].
To evaluate the effect of training set size on model performance, we varied {math}`N^{\text{train size}} \in \{1, 3, 5, 7, 9, 11, 13, 15, 19, 21, 23, 25\}`.

The predictor features for the training and test set were z-scored prior to the regression.
The normalization parameters for z-scoring were only computed from the training set.
This prevented any statistical information leaking from the test data into our evaluation procedure, maintaining the integrity of the cross-validation procedure.

### Performance metrics

The performance of the encoding model was quantified by calculating the Pearson correlation between the observed BOLD responses and the predicted BOLD responses on the held-out test story.
Following @lebel_natural_2023 we averaged the encoding model's performance across 15 repetitions. This allows for a more reliable estimate thereby reducing the risk of performance being biased by any particular split of the data.

## Code

All of our analysis and visualization code was implemented in Python 3.12 and the Python scientific stack (NumPy [@harris_array_2020], Pandas [@mckinney_data_2010; @the_pandas_development_team_pandas-devpandas_2024], SciPy [@virtanen_scipy_2020], scikit-learn [@pedregosa_scikit-learn:_2011], matplotlib [@hunter_matplotlib_2007; @the_matplotlib_development_team_matplotlib_2024], seaborn [@waskom_seaborn_2021], pycortex [@gao_pycortex_2015]).
The code for the replication and reproducibility experiments and its documentation is available at a standalone GitHub repository[^enc_repo].

[^enc_repo]: https://github.com/GabrielKP/enc


# Results

## Model performance with increasing training set size

@lebel_fmri_2023 report that in general test-set performance increases with the increasing training set size (i.e. number of stories used to fit the model) increases. We sought to establish that the same trend holds for our pipeline and the two encoding models. We fit models for $N \in \{1, 3, 5, 7, 9, 11, 12, 15, 20\}$ stories. Results are shown in [](#fig-training-curve). 

The results confirm the increase in performance with more training data, however as noted above, our model undeperforms relative to published results exhibting lower correlation scores over all.


Below, we report results for two fMRI encoding models: based on [distributional word embeddings](#methods-embeddings-model) ('semantic model') and based on [audio envelope](#methods-audio-model) ('sensory model').

## Replication: Semantic encoding model

In [](#fig-embedding)[^icons_attribution] we show the test-set correlation results across the whole brain for participant `UTS02`. The highest performance is achieved with with the largest traininset (20 training stories). The best performing voxels are found in the bilateral temporal, parietal, and prefrontal cortices which is broadly in line with the spatial patterns in the original report [@lebel_fmri_2023]. The best performing voxels showed correlation values of ~0.35, which is lower than in the original report where highest scores reach a correlation of ~0.7. That is, our models pick up on the signal in the relevant brain areas, they are underporfming relative to original results.

```{figure} fig/manuscript_figures/figure1-main.svg
:label: main_figure
**A-C**: Semantic encoding model performance (whole brain average) per participant with increasing training set sizes. Each line shows mean performance (shaded areas show standard error of the mean across 15 repetitions). **A)** Screen capture of figure published by @lebel_natural_2023 [CC BY 4.0], B) reproduction experiment, and **C)** the replication experiment. **D-F:** The results of the semantic encoding model for one participant (S02). The plots show the test-set performance with 25 training stories for each brain voxel on a flattened two-dimensional brain map. **D)** Figures from the original paper, **E)** for the reproduction experiment, and **F)** for the replication experiment.

```

[^icons_attribution]:  Icons from Flaticon.com: https://www.flaticon.com/free-icons/financial-report, https://www.flaticon.com/free-icons/data, https://www.flaticon.com/free-icons/unstructured-data

## Reproducibility experiment: Semantic encoding model

```{figure} #cell.fig.repro.embeddings
:label: fig-embedding

Test-set performance of the embeddings model with different training set sizes. Brigher color-coded voxels indicate better model performance. Test-set performance (Pearson correlation) is averaged across $N = 15$ independent models that were trained by resampling the training set 15 times.
```

## Extension experiment: Sensory encoding model

To additionally our benchmark semantic model results, we implemented a simpler fMRI encoding model based on just the instantaneous envelope of the acoustic energy around word onsets. Our guiding expectation was that such a model would show spatially differnent patterns from from the more complex and statistically powerful semantic encoding model. [](#fig-envelope) displays the results. We see that compared to the semantic model ([](#fig-embedding)), the performance is has a much n arrower spatial extent and predominantly capures signal surrounding the primary auditory cortex (labeled AC) as expected by a low-level sensory model.

```{figure} #cell.fig.extension.envelope
:label: fig-envelope
Test-set performance of the audio encoding model. The peak performance is observed in auditory cortex (AC).
```