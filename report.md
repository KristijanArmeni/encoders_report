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
Time series were available as `.h5p` files for each participant and each story. Following the original report, the first 10 seconds (5 TRs) of each story were trimmed to remove the 10-second silence period before the story began.

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

## Reproduction experiment

We used the original data and code to reproduce the original experimental results.
The code provided the core regression component, but lacked the capability to reproduce the figures.
Thus, we ported the code into our own code base.
To compute the results, we retained the default parameters, except for parameters `nboots`, `chunklen`, and `nchunk`[^parameters] which were set to 20, 10, and 10, respectively (see Table [](#tab:regression_parameters)).

[^parameters]: For an explanation of the parameters see the "Code" section of the original paper [@lebel_natural_2023].

**We successfully reproduced published results.** In [](#main_figure)[^icons_attribution], we show the results of our reproduction experiment (panel B) next to originally published figures (panel A).
Using the shared code and data for the three best performing participants (`S01`, `S02`, `S03`), we were able to reproduce the reported results.
As in the published results, the highest performing participant was `S03` with the highest average test-set correlation at {math}`r \approx 0.08`, followed by `S02` at {math}`\approx 0.07`, and `S01` at {math}`\approx 0.06`.

**Model performance increased with larger training dataset sizes.**
Our reproduction further confirmed that model performance depends on the amount of training data available for each participant.
For the best performing model, performance nearly tripled (from average correlation about 0.03 to 0.09) between the smallest ({math}`N^{\text{stories}} = 1`) and the largest ({math}`N^{\text{stories}} = 25`) training set size.

**Highest performance in the brain areas comprised the language network.** 
The brain language network consists of multiple areas on the left and right sides behind the temples (temporal cortices), the front (frontal cortex) and the upper middle part (parietal cortex) [@friederici_language_2013; @hagoort_neurobiology_2019; @fedorenko_language_2024].
Breaking down the average performance across the brain for each voxel for the best performing model (for participant `S02` at {math}`N^{\text{stories}} = 25`, [](#main_figure), panel E), we found, in line with the original report, that the strongest performance was achieved in the left temporal cortex with a peak correlation of about 0.5, followed by areas in the frontal and parietal cortices.

## Replication experiment

**We replicated the spatial and training set size effects, but not the effect size.** 
The results for model performance on example subject are shown in [](#main_figure), Panel C.
Overall, we managed to reproduce the training set size effects and the broader spatial brain patterns.
In terms of spatial patterns, the best performing voxels were found in the bilateral temporal, parietal, and prefrontal cortices which is broadly in line with the spatial patterns in the original report and our replication.
As in the original report, our reproduction results broadly confirmed that the model showed better performance on the held-out story if trained on a larger set containing more training stories.
However, the effect size obtained with our reproduction pipeline was almost half the size of the original results.
Our best performing model did not go beyond average performance of 0.05 versus 0.09 for the best performing model in the original report and replication.
That is, our models capture the language-related brain activity in the relevant brain areas, but they underperformed compared with the original results.

```{figure} fig/manuscript_figures/figure1-main.svg
:label: main_figure
**A-C**: Semantic encoding model performance (whole brain average) per participant with increasing training set sizes. Each line shows mean performance (shaded areas show standard error of the mean across 15 repetitions). **A)** Screen capture of figure published by @lebel_natural_2023 \[CC BY 4.0\], B) reproduction experiment, and **C)** the replication experiment. **D-F:** The results of the semantic encoding model for one participant (S02). The plots show the test-set performance with 25 training stories for each brain voxel on a flattened two-dimensional brain map. **D)** Figures from the original paper, **E)** for the reproduction experiment, and **F)** for the replication experiment.

```

[^icons_attribution]:  Icons from Flaticon.com: https://www.flaticon.com/free-icons/financial-report, https://www.flaticon.com/free-icons/data, https://www.flaticon.com/free-icons/unstructured-data

### Why did replication results diverge?

To assess the discrepancy between the reproduction and replication results, we incorporated the regression function from the original codebase into our replication pipeline ([](#figure_patching), panels A & B).
This modification substantially improved model performance and successfully recovered the results reported in the original paper and reproduction experiment ([](#figure_patching), Panels C & D).
Although we did not formally test the contributions of differences between implementations, we identified two crucial distinctions in the original code's regression approach that likely explain the performance improvement: 1) the SVD-based implementation explicitly removed components of the regressor matrix with small singular values; 2) instead of standard k-fold cross-validation, a chunking with bootstrapping strategy [@huth_natural_2016] was employed for hyperparameter optimization.

```{figure} fig/manuscript_figures/figure2-patching.svg
:label: figure_patching
:width: 60%

Replication results were improved after patching our code with regression implementation from @lebel_natural_2023. **A)** The pipeline of our replication used the regression function in [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html) [@pedregosa_scikit-learn:_2011]. **B)** The 'patched' pipeline used the regression function from [Lebel et al.](https://github.com/HuthLab/deep-fMRI-dataset/blob/master/encoding/ridge_utils/ridge.py) **C & D:** Semantic encoding model performance (whole brain average) per participant with increasing training set sizes. Each line shows mean performance (shaded areas show standard error of the mean across 15 repetitions).
```

## Extension: Sensory encoding model

One of the core motivations behind reproducible computational research is that it should, in principle, allow researchers to easily extend and build on the shared work.
To further benchmark our replication pipeline, we explored the feasibility of performing a small but meaningful extension to the original semantic encoding model.
Processing of semantic features in language input is a high-level cognitive process that engages distributed brain networks [@binder_where_2009; @huth_natural_2016].
As a contrast, we implemented a simpler encoding model based solely on a single one-dimensional acoustic feature of the stimulus, namely, instantaneous fluctuations of the auditory envelope which we termed sensory model.

In [](#figure_extension) we report the results of the sensory model.
Compared to the semantic model, it captured activity in a more restricted set of brain areas, with peak performance localized to the auditory cortex (AC) – a sensory area involved in early-stage auditory processing.
This observation is broadly consistent with prior work, which has repeatedly shown that activity in auditory cortical regions is strongly modulated by low-level acoustic features, while higher-order language processing recruits a more extensive group of regions including temporal, parietal, and frontal cortices [@hickok_cortical_2007].

```{figure} fig/manuscript_figures/figure3-extension.svg
:label: figure_extension

Sensory model encoding performance showed a narrower set of regions compared to the semantic model. **A)** Acoustic encoding model performance for each participant with increasing training set size (intact and shuffled predictors, shaded areas show standard error of the mean across 15 repetitions). **B)** Voxel-specific performance for `S02`.
```