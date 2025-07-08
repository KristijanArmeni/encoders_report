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
abbreviations:
  MyST: Markedly Structured Text
  HPC: High Performance Computing
bibliography: library.bib
---


# Introduction

Computational methods are an indispensable part of contemporary science.
For example, cognitive computational neuroscience is a nascent discipline using computational models to investigate the brain's representations, algorithms, and mechanisms underlying human intelligent behavior in healthy and clinical populations [@kriegeskorte_cognitive_2018; @naselaris_cognitive_2018].
As computational techniques become central to scientific discovery, the ability of independent research teams to reproduce and replicate published work has been recognized as a standard for evaluating the reliability of scientific findings [@claerbout_electronic_1992; @donoho_invitation_2010; @peng_reproducible_2009; @national_academies_of_sciences_engineering_and_medicine_reproducibility_2019].
Researchers are now strongly encouraged to share open and reproducible research data, code, and documentation alongside written reports [@peng_reproducible_2011; @pineau_improving_2021; @jwa_spectrum_2022; @nelson_ensuring_2022], a trend also reflected in cognitive computational neuroscience [@sejnowski_putting_2014; @poldrack_making_2014; @gorgolewski_brain_2016; @poldrack_scanning_2017; @botvinik-nezer_reproducibility_2023; @de_vries_sharing_2023].
Despite this progress, achieving computational reproducibility in research remains a significant practical challenge, due to the increasing complexity of both data and analytical methods, as well as ethical and legal constraints [e.g., @open_science_collaboration_estimating_2015; @botvinik-nezer_variability_2020; @jwa_spectrum_2022; @errington_challenges_2021; @hardwicke_analytic_2021; @xiong_state_2023; @barba_path_2024].

While there is growing momentum toward sharing research artifacts such as data and code, making them available does not guarantee reproducibility [@mckiernan_policy_2023].
To be truly reproducible, research artifacts must also be reusable, readable, and resilient [@connolly_software_2023].
Reproduction efforts can falter when artifacts are incomplete or contain errors -- such as missing data or faulty scripts [e.g., @pimentel_large-scale_2019; @hardwicke_analytic_2021; @nguyen_are_2025] -- or when the documentation is sparse or unclear, making it difficult to rerun analyses with the provided data [@xiong_state_2023].
In contrast, the software engineering community has long developed best practices for creating robust and reusable general-purpose artifacts [e.g., @thomas_pragmatic_2020; @beck_manifesto_2001; @martin_clean_2012; @washizaki_guide_2024].
However, these practices are not always directly applicable to research settings, where the scope of engineering needs varies from standalone analysis scripts to domain-specific research software tools [@balaban_ten_2021; @connolly_software_2023].
Given that much of scientific research is conducted within relatively small, independent teams, a central question arises: what software practices and open-source tools are best suited to support reproducible research?

Here, we share our experience and lessons learned from a small-scale reproducibility project in an academic neuroscience lab.
We set out to reproduce and replicate the evaluation of predictive models on a publicly available fMRI dataset, which demonstrated that increasing the amount of training data improved average performance in predicting brain activity at the individual participant level [@lebel_natural_2023].
We conducted three complementary experiments: i) a **reproduction** experiment using the original shared data and code, ii) a **replication** experiment using the shared data but re-implementing the analysis with our own code, and iii) an **extension** experiment using the shared data in a novel analysis not reported in the original paper.

The main contributions of our work are:

1. We successfully reproduced the original results using the shared data and code.
2. We identified several challenges when replicating the analysis with our own implementation.
3. We describe components of our reproducible workflow and review software engineering practices that can facilitate reproducible research.

# Methods

In what follows, we provide a high-level overview of the methodology, the data used, and the analysis design for our experiments following the original experiment and dataset [@lebel_natural_2023].
All of our analysis and visualization code was implemented in Python 3.12 and the Python scientific stack (NumPy [@harris_array_2020], Pandas [@mckinney_data_2010; @the_pandas_development_team_pandas-devpandas_2024], SciPy [@virtanen_scipy_2020], scikit-learn [@pedregosa_scikit-learn:_2011], matplotlib [@hunter_matplotlib_2007; @the_matplotlib_development_team_matplotlib_2024], seaborn [@waskom_seaborn_2021], pycortex [@gao_pycortex_2015]).
The data and code for the replication and reproducibility experiments and the documentation is available at standalone OSF[^osf] and GitHub repositories[^github], respectively.

[^osf]: https://osf.io/8w5cb/
[^github]: https://github.com/GabrielKP/enc

## Encoding models

A central goal in computational neuroscience is to test hypotheses about the computations and representations the brain uses to process information.
Encoding models address this goal by quantifying the relationship between features of a stimulus and its corresponding neural responses.

In practice, machine learning methods are often used to extract relevant stimulus features (e.g., semantic, visual, or acoustic), which are then used in statistical models (typically linear regression [@ivanova_beyond_2022]) to predict neural activity [@naselaris_encoding_2011]. By comparing the predictive performance of different encoding models (e.g., from different stimulus features across different brain areas), researchers can draw inferences about the spatial and temporal organization of brain function [@kriegeskorte_interpreting_2019; @doerig_neuroconnectionist_2023]. Encoding models are also foundational for brain-computer interfaces, including clinical applications such as speech prostheses, which aim to reconstruct intended speech from recorded brain activity [@silva_speech_2024].

Formally, an encoding model estimates a function that maps stimulus features $\mathbf{X}$ to brain responses $\mathbf{Y}$. The model typically takes a linear form:

```{math}
\hat{\mathbf{Y}} = \mathbf{X} \hat{\mathbf{W}}
```

where {math}`\hat{\mathbf{Y}} \in \mathbb{R}^{T \times N}` denotes predicted brain responses across $T$ time points and $N$ voxels, and {math}`\mathbf{X} \in \mathbb{R}^{T \times D}` contains $D$-dimensional stimulus features. The weight matrix {math}`\hat{\mathbf{W}} \in \mathbb{R}^{D \times N}` captures the contribution of each feature dimension to each voxel’s activity.

Model estimation involves fitting $\hat{\mathbf{W}}$ to predict brain activity $\hat{\mathbf{Y}}_{\text{train}}$ from features $\mathbf{X}_{\text{train}}$ using statistical methods such as ridge regression. Model performance is then evaluated on held-out test data by comparing predicted brain activity $\hat{\mathbf{Y}}_{\textsf{test}}$ to observed brain activity $\mathbf{Y}_{\textsf{test}}$ using a scoring function such as Pearson correlation: $r = \textsf{correlation}(\hat{\mathbf{Y}}_{\textsf{test}}, \mathbf{Y}_{\textsf{test}})$.

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

### Preprocessed fMRI responses ({math}`\mathbf{Y}_{fmri}`)

Because the focus of our work was on encoding models, we chose to use the preprocessed fMRI data provided by @lebel_natural_2023.
As a result, our reproducibility efforts did not involve preprocessing of raw fMRI data, which includes steps such as motion correction, cross-run alignment, and temporal filtering.
We limited our analysis to a subset of three participants (`S01`, `S02`, `S03`) because they showed the highest signal quality and highest final performance in the original paper.
We reasoned that this data would be sufficient to evaluate reproducibility and replicability, as higher quality data would allow us to make more nuanced evaluations in case we found performance discrepancies, and that this data is more likely to be used by other researchers.

We accessed the data using the DataLad data management tool [@halchenko_datalad_2021], as recommended by @lebel_natural_2023.
Time series were available as `.h5p` files for each participant and each story. Following the original report, the first 10 seconds (5 {term}`TR`s) of each story were trimmed to remove the 10-second silence period before the story began.

**Hemodynamic response estimation**  
The fMRI BOLD responses are thought to represent temporally delayed (on the scale of seconds) and slowly fluctuating components of the underlying local neural activity [@logothetis_underpinnings_2003].
Following @lebel_natural_2023 we constructed predictor features for timepoint {math}`t` by concatenating stimulus features from time points {math}`t - 1` to {math}`t - 4` ({math}`n_{delay}=4`) to account for the delayed neural response.
For timepoints where {math}`t-k<0`, zero vectors were used as padding.
This resulted in a predictor matrix {math}`X \in \mathbb{R}^{T\times 4D}`.
Although this increases computational cost, it enables the regression model to capture the shape of the hemodynamic response function [@boynton_linear_1996] underlying the BOLD signal.


## Predictors

### Semantic predictor ({math}`\mathbf{X}_{\textsf{semantic}}`)

To model brain activity related to aspects of linguistic meaning understanding during story listening, @lebel_natural_2023 used word embeddings --- high-dimensional vectors capturing distributional semantic properties of words based on their co-occurrences in large collections of text [@clark_vector_2015].

**Extracting word embeddings and timings** For each word in the story, we extracted its precomputed 985-dimensional embedding vector [@huth_natural_2016] from the `english1000sm.hf5`[^english1000sm] data matrix (a lookup table) provided by @lebel_natural_2023 and available in the OpenNeuro repository.
Words not present in the vocabulary were assigned a zero vector.
For every story, this yielded a {math}`\hat{\mathbf{X}}_{semantic} \in \mathbb{R}^{N_{words} \times 985}` matrix of word embeddings for each story where {math}`N_{words}` are all individual words in a story.
The onset and offset times (in seconds) of words were extracted from `*.TextGrid` annotation files[^textgrid] provided with the original dataset.

[^english1000sm]: https://github.com/OpenNeuroDatasets/ds003020/blob/main/derivative/english1000sm.hf5

[^textgrid]: These are structured .txt files that are used with the Praat software for acoustic analysis (https://www.fon.hum.uva.nl/praat/)

**Aligning word embeddings with fMRI signal.** The fMRI BOLD signal was sampled at regular intervals (repetition time, or TR = 2s). To compute a stimulus matrix {math}`\hat{\mathbf{X}}_{semantic}` that matched the sampling rate of the BOLD data, following @lebel_natural_2023, we first constructed an array of word times {math}`T_{word}` by assigning each word a time half-way between its onset and offset time. This was used to transform the embedding matrix (which was at discrete word times) into a continuous-time representation. This representation is zero at all timepoints except for the middle of each word {math}`T_{word}`, where it is equal to the embedding vector of the word. We then convolved this signal with a Lanczos kernel (with parameter {math}`a=3` and {math}`f_{cutoff}=0.25` Hz) to smooth the embeddings over time and mitigate high-frequency noise. Finally, we resampled the signal half-way between the TR times of the fMRI data to create the feature matrix used for regression, {math}`\mathbf{X}_{semantic} \in \mathbb{R}^{N_{TRs} \times N_{dim}}`.

### Sensory predictor ({math}`\mathbf{X}_{\textsf{sensory}}`)

To benchmark the predictive performance of our semantic predictor, we decided to develop an additional, simpler encoding model based on acoustic properties of the stories (not reported in the original work by @lebel_natural_2023).
The audio envelope was computed by taking the absolute value of the hilbert-transformed wavfile data.
For each story, the envelope was trimmed at the end by dropping the 10 beginning and final seconds.
The trimmed envelope was then downsampled to the sampling frequency of the fMRI data.

## Ridge regression

In the reproduction experiment, we used the ridge regression implementation provided with the shared code. 
In our replication experiment, we used the scikit-learn library [@pedregosa_scikit-learn:_2011], which is a mature library for machine learning with a large user-base and community support.
Specifically, we used the `RidgeCV()` class which performs a leave-one-out cross-validation to select the best value of the {math}`\alpha` hyperparameter for each target variable (i.e. brain activity time courses in each voxel) prior to fitting the model.
We set the possible hyperparamater values to {math}`\alpha \in` `np.logspace(1, 3, 10)` as in the original report [@lebel_natural_2023].
The {math}`\alpha` value that resulted in the highest product-moment correlation coefficient was then used by `RidgeCV()` as the hyperparameter value to fit the model.

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

The performance of the encoding model was quantified as the Pearson correlation between observed and predicted BOLD responses on the held-out test story.
To obtain a more reliable estimate, we averaged encoding model performance across 15 repetitions, thereby reducing the risk of performance being biased by any particular split of the data.

:::{table}
:label: regression_table
![](#nb_regression_table)

Regression parameters as used in the original work, reproduction, and replication experiments. For explanation of individual parameters see main text.
:::

## Code

All of our analysis and visualization code was implemented in Python 3.12 and the Python scientific stack (NumPy [@harris_array_2020], Pandas [@mckinney_data_2010; @the_pandas_development_team_pandas-devpandas_2024], SciPy [@virtanen_scipy_2020], scikit-learn [@pedregosa_scikit-learn:_2011], matplotlib [@hunter_matplotlib_2007; @the_matplotlib_development_team_matplotlib_2024], seaborn [@waskom_seaborn_2021], pycortex [@gao_pycortex_2015]).
The code for the replication and reproducibility experiments and its documentation is available at a standalone GitHub repository[^enc_repo].

[^enc_repo]: https://github.com/GabrielKP/enc



# Results

## Reproduction experiment

We used the original data and code to reproduce the experimental results.
Since the provided code contained the core regression component but lacked the capability to generate figures, we integrated it into our own codebase.
We retained the default parameters, except for parameters `nboots`, `chunklen`, and `nchunk`[^parameters] which were set to 20, 10, and 10, respectively (see [](#regression_table)).

[^parameters]: For an explanation of the parameters see the "Code" section of the original paper [@lebel_natural_2023].

**We successfully reproduced published results.** [](#main_figure)[^icons_attribution] shows results of our reproduction experiment (panel B) alongside the originally published figures (panel A).
Using the shared code and data for the three best performing participants (`S01`, `S02`, `S03`), we reproduced the reported results with participant `S03` showing the highest average test-set correlation at {math}`r \approx 0.08`, followed by `S02` at {math}`\approx 0.07`, and `S01` at {math}`\approx 0.06`.

**Model performance increased with larger training dataset sizes.** Our reproduction further confirmed that model performance depends on the amount of training data.
For the best performing model, average performance nearly tripled (from 0.03 to 0.09) between the smallest ({math}`N^{\text{stories}} = 1`) and the largest ({math}`N^{\text{stories}} = 25`) training sets.

**Highest performance in the brain areas comprised the language network.** Examining the spatial distribution of model performance across individual voxels, we expected the highest performance to occur in regions known to process language.
The language network spans multiple lobes, including bilateral temporal, frontal, and parietal cortices [@friederici_language_2013; @hagoort_neurobiology_2019; @fedorenko_language_2024].
For the best performing model (for participant `S02` at {math}`N^{\text{stories}} = 25`, [](#main_figure), panel E), peak performance was achieved for voxels in the left temporal cortex (correlation approximately 0.5), followed by frontal and parietal regions, consistent with the original report.

## Replication experiment

**We replicated the spatial and training set size effects, but not the effect size.** The results for an example subject are shown in [](#main_figure), Panel C.
We successfully reproduced the training set size effect and the overall spatial pattern.
The best performing voxels were found in bilateral temporal, parietal, and prefrontal cortices, consistent with the original report and our replication.
As in the original report, our reproduction results broadly confirmed that the model trained on larger datasets (i.e., more stories) showed better performance on the held-out story.
However, our reproduction pipeline yielded substantially lower effect sizes than the original results.
Our best performing model achieved only 0.05 average performance compared to 0.09 in the original report and replication.
While our models captured the language-related brain activity in relevant regions, they underperformed relative to the original results.

:::{hint}
If parts of figures are not rendering correctly (can happen with panels containing brain plots), try refreshing the browser window.
:::

```{figure} fig/manuscript_figures/figure1-main.svg
:label: main_figure
**A-C**: Semantic encoding model performance (whole brain average) per participant with increasing training set sizes. Each line shows mean performance (shaded areas show standard error of the mean across 15 repetitions). **A)** Screen capture of figure published by @lebel_natural_2023 \[CC BY 4.0\], B) reproduction experiment, and **C)** the replication experiment. **D-F:** The results of the semantic encoding model for one participant (S02). The plots show the test-set performance with 25 training stories for each brain voxel on a flattened two-dimensional brain map. **D)** Figures from the original paper, **E)** for the reproduction experiment, and **F)** for the replication experiment.

```

[^icons_attribution]:  Icons from Flaticon.com: https://www.flaticon.com/free-icons/financial-report, https://www.flaticon.com/free-icons/data, https://www.flaticon.com/free-icons/unstructured-data

### Why did replication results diverge?

We identified two key distinctions between the original code and our implementation: 1) instead of standard k-fold cross-validation, the original code used a chunking and bootstrapping strategy [@huth_natural_2016] for hyperparameter optimization; 2) singular value decomposition (SVD) was applied to remove low-variance components from the regressor matrix prior to regression.
To isolate the effects of these differences, we selectively incorporated elements from the original code into our replication pipeline ([](#figure_patching), panel A) and compared model performance against our replication results ([](#figure_patching), panel B).
We found that incorporating only the chunking and bootstrapping strategy did not improve performance ([](#figure_patching), panel C).
In contrast, fitting ridge regression after SVD --- using the function from the original code --- restored model performance to the level reported in the original paper ([](#figure_patching), panel D).
These results suggest that the custom ridge regression implementation was essential for reproducing the original results.

```{figure} fig/manuscript_figures/figure2-patching.svg
:label: figure_patching
:width: 100%

Replication results were improved after patching our code with regression implementation from @lebel_natural_2023. **A)** The pipeline of our replication used the regression function in [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html) [@pedregosa_scikit-learn:_2011]. **B)** The 'patched' pipeline used the regression function from [Lebel et al.](https://github.com/HuthLab/deep-fMRI-dataset/blob/master/encoding/ridge_utils/ridge.py) **C & D:** Semantic encoding model performance (whole brain average) per participant with increasing training set sizes. Each line shows mean performance (shaded areas show standard error of the mean across 15 repetitions).
```

## Extension: Sensory encoding model

One of the core motivations behind reproducible computational research is that it should, in principle, allow researchers to extend and build on the shared work.
To further benchmark our replication pipeline, we explored the feasibility of performing a small but meaningful extension to the original semantic encoding model.

Processing of semantic features in language input is considered a high-level cognitive process that engages a broadly distributed set of brain regions [@binder_where_2009; @huth_natural_2016].
Next to the semantic interpretation of speech, a narrower set of brain areas near the temples, known as superior temporal cortex, processes speech in terms of its low-level acoustic information (e.g., phonemes, syllables) [@hickok_cortical_2007; @obleser_pre-lexical_2009; @price_anatomy_2010].
To model this, we implemented a simpler encoding model based solely on a single one-dimensional acoustic feature of the stimulus, namely, instantaneous fluctuations of the audio envelope (see Section {ref}`sensory_predictor`) which we termed "sensory model".

In this experiment we did not have prior results on the same dataset to benchmark against.
Instead, we constructed a simple control condition by randomly shuffling the target variable (BOLD signals) across time and repeating the regression fitting procedure.
The performance of the shuffled regression showed virtually no correlation with fMRI data ("x" markers in [](#figure_extension)) suggesting that sensory models uncovered meaningful neural responses to the audio envelope.

Overall, the results of the sensory model ([](#figure_extension)) are broadly consistent with prior work.
Not only did the model perform well above the shuffled baseline, it also captured activity in a restricted set of brain areas, with peak performance localized to the auditory cortex in both hemispheres.
Prior work has reliably shown that low-level acoustic features strongly modulate activity localized to auditory cortical regions [@hickok_cortical_2007; @obleser_pre-lexical_2009; @price_anatomy_2010].

```{figure} fig/manuscript_figures/figure3-extension.svg
:label: figure_extension

Sensory model encoding performance showed a narrower set of regions compared to the semantic model. **A)** Acoustic encoding model performance for each participant with increasing training set size (intact and shuffled predictors, shaded areas show standard error of the mean across 15 repetitions). **B)** Voxel-specific performance for `S02`.
```

# Discussion

Reproducible computational science ensures scientific integrity and allows researchers to efficiently build upon past work.
Here, we report the results of a reproduction and replication project in computational neuroscience, a scientific discipline with computational techniques at its core [@kriegeskorte_cognitive_2018].
We set out to reproduce and replicate evaluation results of a publicly available neuroimaging dataset recorded while participants listened to short stories [@lebel_natural_2023].
Using the code shared with the paper, we reproduced the results showing that predicting brain activity with semantic embeddings of words in stories improved with increasing size of training datasets in each participant.
When attempting to replicate the results by implementing the original analysis with our own code, our model performance was much lower than the published results.
Accessing the original code, we were able to confirm that the discrepancy was due to the differences in implementation of the regression function.

## Reproduction was largely frictionless, replication and extension were not

There were several examples of good practices that made our reproduction and replications attempts easier.
First, @lebel_natural_2023, distributed their dataset through a dedicated repository for data sharing [@markiewicz_openneuro_2021] and structured the dataset following a domain-specific community standard [@poldrack_making_2014] which made it possible to use a data management software [@halchenko_datalad_2021] for accessing it.
While this might appear a trivial note given that the original work is a dataset descriptor with data sharing as it core objective, it warrants an emphasis as its benefits apply to any empirical work.
Specifically, their approach made it easy for us to access the dataset programmatically when building the pipeline (e.g., writing a single script for downloading the data needed in our experiments as opposed to accessing the data interactively).
This highlights the importance of data management infrastructure and tools in writing reproducible research code.

Second, the authors provided cursory documentation and instructions on how to use their code for reproducing the results.
While the shared code missed some elements that would have been helpful, for example the scripts used for executing the original analyses and the code that produced the figures, the information provided was nevertheless sufficient for us to be able to reproduce their full analysis on the three best participants.
In addition, the provided code was modular, with specific analysis steps being implemented as separate routines (for example, the regression module contained the regression fitting functions etc.), making porting specific modules to our code easy and convenient.
Analyses in computational neuroscience frequently require custom implementations and workflows, which is common in computational research [@balaban_ten_2021], making a single-analysis-script approach impractical.
Navigating the code without basic documentation and sensible organization would have rendered replication much more effortful.

Whereas reproducing the results was mostly achievable, replicating the work with our own code and adapting it for a novel experiment was not without friction.
This came to the fore once we started building and evaluating our replication pipeline from the descriptions in the report.
For example, when implementing the encoding model, we decided to use a standard machine learning library that conveniently implements ridge regression fitting routines, as it was not apparent from the original paper alone to us which kind of ridge regression was needed.
Despite attempting to match the hyperparameter selection procedure, cross-validation scheme, and scoring functions, our encoding models kept underperforming relative to published results.
Until we patched our pipeline with the original regression functions, it was unclear just which aspect of our pipeline was causing lower performance.
Thus, without the shared code it would have been near impossible for us to troubleshoot our process on the basis of the published report alone. 

As another case in point, we encountered challenges when attempting to build on the work and perform a nominally straightforward extension of the original experiment by building an encoding model based on the audio envelope of the stimuli.
Whereas the process to temporally align word embeddings and brain data was documented, following the same recipe to align the shared auditory stimuli and brain signal resulted in inconsistencies in the sizes of the to-be-aligned data arrays.
We were left to perform our best guess as to the proper alignment by triangulating between stimulus length and number of samples, significantly increasing the effort for an otherwise straightforward extension.

## Tallying software engineering practices for reproducibility in neuroscience

Informed by our reproducibility and replication experience, we argue that software engineering practices, such as code documentation, code review, version control, and code testing, among others, are essential in mitigating barriers to reproducibility beyond what is achievable with data and code sharing alone.
In doing so, we rejoin other scientists and software engineers in calling for software engineering maturity in research [e.g., @wilson_wheres_2006; @sandve_ten_2013; @wilson_best_2014; @wilson_good_2017; @storer_bridging_2017; @balaban_ten_2021; @barba_path_2024; @johnson_sciops_2024].
While the benefits conferred by software engineering practices might appear uncontentious _in principle_, adoption of any new practice, be it for an individual or in a larger community, is far from straightforward _in practice_.
Absent appropriate incentives, infrastructure, or norms, adopting a new skill or behavior includes a perceived cost (e.g., the time it takes to acquire a new skill) which can be a significant deterrent to adoption [@nosek_strategy_2019; @armeni_towards_2021].
Software engineering practices are no exception: practices will differ in terms of how difficult they are to adopt (i.e., technical overhead required), the time it takes to apply the practice, and the gain they bring for the researcher or the team adopting them.

:::{table}
:label: practices_table
![](#nb_practices_table)

Our tally of software engineering practices in research with respect to reproducibility gains (the 3R framework by @connolly_software_2023), possible costs, and example tools we adopted.
:::

[](#practices_table) summarizes our own experience in adopting some of the research software practices, how we believe they mitigate the barriers to reproducibility, and their possible costs.
In tallying the reproducibility gains, we follow the framework of 3Rs for academic research software proposed by Connolly et al.: research code should be readable, reusable, and resilient [@connolly_software_2023].
In terms of possible costs, "Technical Overhead" in [](#practices_table) refers to the degree of novel technical expertise required to adopt a practice.
For example, even for someone who regularly writes analysis code, code packaging requires learning the packaging tools, configuration options, the package publishing ecosystem,  etc.
"Time Investment", on the other hand, refers to the amount of time it takes to learn and subsequently apply a practice, once learned.[^practice_costs]

[^practice_costs]: While these two aspects are certainly positively correlated to a significant extent, a practice can require little technical overhead, while still representing non-negligible time investment (e.g. writing documentation) or vice-versa (e.g. code packaging, which is set up only once, but requires dedicated technical know-how).

Research software specifically can vary in scope on a spectrum between a standalone analysis script and up to a mature software project used by a larger research community [@connolly_software_2023].
Research software in computational neuroscience typically falls in between the two extremes.
Computational neuroscience is a diverse discipline in terms of the scale of investigations (e.g., ranging from single-neuron to whole-brain recordings) and the type of measurements used (e.g., static anatomical images, dynamic multi-channel activity over time, or a combination of different data modalities) [@sejnowski_putting_2014].
What the analyses have in common, however, is that they are composed of several stages (e.g., preprocessing, model fitting, statistical inference, visualization)[@gilmore_progress_2017] such that they frequently require custom routines and workflows [@balaban_ten_2021].
With his context in mind, what aspects of software engineering practices can specifically benefit computational reproducibility in human neuroscience?

**Improving readability: Documentation, code formatting, and code review** 

Our replication experience showed that methodological details described in the published report were insufficient in detail and our success hinged on our ability to reverse engineer the procedures with the help of shared code.
Details matter: given the complexity of analytical pipelines in computational neuroscience, even a nominally small analytical deviation can have large cumulative effects on the results [@carp_secret_2012; @gilmore_progress_2017; @poldrack_scanning_2017; @errington_challenges_2021].
Insufficient reporting standards in fMRI research have been confirmed in large-scale analyses of published work [@carp_secret_2012; @guo_reporting_2014; @poldrack_scanning_2017].
Our first set of recommended practices thus centers on code transparency and readability.

Sufficiently documented code significantly boosted our understanding of its purpose and thus our ability to reuse it in replication experiments.
Conversely, the lack of such documentation made even the well-structured code difficult to understand and compelled us to perform a line-by-line walkthrough in order to understand the operations on different variables.
Given the modular nature of shared code in neuroscience [@gilmore_progress_2017], we argue that, apart from inline comments and cursory mentions in papers, all shared code should contain docstrings for major functions and classes and illustrative guides in how they can be invoked.
This is particularly necessary for any custom in-house developed code.

[^mkdocstrings]: https://mkdocstrings.github.io/

Formatting and linting are simple, yet effective methods to improve code readability and to facilitate reusability.
Once set up, dedicated code formatters and linters can be used without additional time investment, but deliver substantial benefits: improved code readability, early detection of potentially costly syntax errors, and style consistency.
This facilitates collaboration both within and across research teams.
The increased readability is not only valuable in collaborative environments, but also in situations were a single researcher reuses to their own code after an extended period.
Overall, we recommend that code formatting and linters become a standard practice in research software development.

Building analysis pipelines in neuroscience is seldom straightforward and often requires frequent iteration cycles.
This can lead to technical debt and poorly organized code with unused parts which renders it difficult to understand [@vliet_seven_2020].
Code review, a practice of auditing the code for style, quality, and accuracy by peers or developers other than the authors [@ackerman_software_1989; @bacchelli_expectations_2013], can help mitigating the accrued technical debt.
Systematic quality control is a sign of operational maturity in scientific teams dealing with complex analysis projects [@johnson_sciops_2024].
Apart from quality control, code review brings other important benefits such as increased author confidence, collegiality, and cross-team learning [@vable_code_2021; @rokem_ten_2024].
We conducted code reviews in the form of brief code walkthroughs after every meeting.
Incremental adoption was crucial.
Reviewing entire research code can be burdensome, reviewing smaller code chunks, soon after we implemented them, was very doable, improved code clarity, and on multiple occasions allowed us to catch mistakes early.
Like documentation, code review is a relatively straightforward practice to adopt insofar that it requires minimal to no technical overhead.

**Improving resilience and reuse: Version control, packaging, and testing**
We view version control, packaging, and code testing as practices that require greater investment in familiarity with technical tools and possibly steeper learning curves.
Version control, a systematic way of recording changes made to files over time [@community_turing_2022], is a standard practice in software engineering.
While version control does not directly contribute to readability or reusability (in the sense that even extensively versioned code may not be re-executable), it is essential in making the computational research process resilient --- in the event of catastrophic changes, current state of files can be reverted to a previous state.
We adopted version control in a simple collaborative workflow where we all team members had write access to a joint remote repository.

Reproducible code guarantees that independent researchers can obtain the original results.
But that does not imply that the shared code is straightforward to reuse.
Reusable code is easy to install and set up, which facilitates adoption [@ma_human_2024], and is part of the FAIR principles of research software [@chue_hong_fair_2022].
Depending on complexity, research code can be made reusable in various ways [@community_turing_2022], one option is to distribute it as an installable _package_: a collection containing the code to be installed, specification of required dependencies, and any software metadata (e.g., author information, project description, etc.).
In practice, packaging means organizing your code following an expected directory structure and file naming conventions [@wasser_pyopenscipython-package-guide_2024].

In addition to facilitating code reuse by independent researchers, one of the advantages of installable code is that it can be reused within and across your own projects.
That turned out to be a crucial design factor in our reproducible workflow (see Section {ref}`sec:workflow` below), where we wrapped our figure-making code into separate functions within the package.
These functions can be invoked as part of standalone scripts or imported in interactive notebooks and computational reports.
The separation between analysis code (package) and its subsequent reuse (in computational notebooks, reports) follows the principle of modular code design, also known as _do-not-repeat-yourself_ (DRY) principle [@wilson_good_2017].
Most programming languages come with dedicated packaging managers [@alser_packaging_2024].
Whereas Python packaging ecosystem is sometimes perceived as unwieldy[^xkcd_packaging], we found that modern packaging tools, for example Poetry[^poetry] and uv[^uv], were mostly straightforward to use, did not incur substantial technical overhead, and required moderate time investment (e.g., {math}`\sim 60` mins).

[^xkcd_packaging]: https://xkcd.com/1987
[^poetry]: https://python-poetry.org/
[^uv]: https://docs.astral.sh/uv/

Finally, possibly the most extensive practice we include on our list is code testing.
Code testing is a process of writing dedicated routines that test specific parts of code or entire workflows for accuracy and syntactic correctness.
Test-driven development in pure software is a mature discipline [@beck_test-driven_2015].
It provides the broadest reproducibility benefits, yet testing research code comes with its own specific considerations and challenges [@eisty_testing_2025].
To highlight just two, analysis code in empirical research depends on the data which can contain inconsistencies and exceptions making it challenging to write a single test covering all exceptions.
In addition, in neuroscience the datasets tend to be large (on the orders of tens or hundreds of gigabytes) and given the need for tests to be performed frequently, dedicated test datasets would need to be created that appropriately mock real dataset characteristics while remaining lightweight.
Second, the research process is iterative (e.g., output of one analysis stage shapes the analysis at the next stage) and the boundaries between the development and deployment phases are frequently blurry [@vliet_seven_2020], leading to challenging testing decisions and frequent need for updating the test suite.
Developing testing code demands upfront knowledge of requirements and substantial investment of time and dedicated software expertise, neither of which are currently plentiful in neuroscience.

Testing is the final software engineering practice on our list for a reason; we only started writing limited unit tests and basic package installation tests later in the project phase, once the workflows matured and required less frequent changes.
Despite the challenges, we see clear benefits in striving towards test-driven development in research as it forces researchers to not only ascertain accuracy of code, but also to to think carefully about code design, and render explicit code assumptions.

## Towards reproducible scientific publishing workflows {#sec:workflow}

[](#figure_workflow) shows a high-level overview of our adopted code, documentation, and publishing workflow.
It shows three main streams: i) packaged code and documentation, ii) computation, storage, and archiving, and iii) computational report.
This architecture allows research code (e.g., analysis and plotting scripts) to be developed and versioned in one place and independently of the deployment (HPC) and the publishing streams.
The workflow is semi-automatic.
For example, documentation and report are deployed and rebuilt automatically upon changes to repositories via GitHub actions and GitHub Pages.
However, if analyses change, the researcher must redeploy the jobs on the computing cluster, download the new results, re-execute the report pipeline, and update the remote repositories.

:::{figure} fig/manuscript_figures/figure4-workflow.svg
:label: figure_workflow
:width: 70%

A schematic overview of our software development and publishing workflow.
:::

**Package and documentation** The analysis package contains the separate `.py` modules corresponding to distinct analysis stages (e.g., `data.py` for loading the data, `regression.py` for model fitting, etc.) which provide the relevant functions (e.g., `data.load_fmri`, `regression.ridge_regression`, etc.).
The analysis code is locally installable as a python package (via  Poetry[^poetry]) meaning that after downloading, the user can install the code and dependencies, for example using either `poetry install` or via `pip install -e`.
The code was versioned using git and hosted on GitHub, licensed with permissive MIT License.

We used MkDocs[^mkdocs] to parse the contents or the `./docs` folder in the project repository and to render the documentation website.
We used MkDocstrings extension that automatically parsed analysis code docstrings and included it as part of the documentation website.
The documentation site was published via GitHub Pages and set to redeploy automatically upon updates to the code repository via a GitHub actions.

[^mkdocs]: https://www.mkdocs.org/

:::{code} python
:label:code_example
:linenos:
:caption: Example usage of packaged research code. The code for figures from the analysis package is imported and exectued in a jupyter notebook. The user is expected to have downloaded precomputed data (model scores) beforehand. The notebook cell rendering the figure is labeled and the figure can be reused in a MyST interactive report.

# import code for figures from the package
from encoders.plots import make_brain_figure

#| label:brainfig
fig = make_brain_figure(
    data_folder="/path/to/downloaded/data"
)

:::

**Data and computation** All our analyses were deployed on a high-performance computing (HPC) cluster.
Because the encoding models are fit across the entire brain (resulting in $\approx10^3$ target variables in a regression model) using high-dimensional predictors (e.g., frequently several hundred dimensions), they require sufficiently powered computing infrastructure.
For example, to fit a model on the largest training set size with 25 stories, we requested about 60 gigabytes of memory (RAM).
This required fitting multiple models in parallel in a separate deployment stream and precluded, for example, executing all the analyses sequentially in an interactive jupyter notebook session.

**Publishing** A separate repository was used to visualize the results and publish an interactive research report[^enc_report] using the MyST Markdown framework [@rowan_cockett_jupyter-bookmystmd_2025].
MyST allows users to author a computational report in a markdown file, providing all the functionality for technical writing (e.g., citations, cross-references, math rendering, etc.).
A report file (`report.md`) is paired with a jupyter notebook which imports the `plots.py` module and reuses the code that creates the figures.
Because the notebook cells outputting the figures are labeled (following MyST syntax, see [](#code_example)), they can be referenced and reused in the report.

[^enc_report]: https://kristijanarmeni.github.io/encoders_report/

[^mystmd]: https://mystmd.org/

## Limitations

**End-to-end reproducibility** From the perspective of reproducibility, our workflow falls short in certain respects.
Fitting all encoding models reported in this work necessitates access to a high-performance computing (HPC) cluster[^hpc_access], which is not universally available to all research institutions or individual researchers.
However, researchers without access to HPC or substantial parallelism can still reproduce the figures from precomputed results (i.e., encoding model performance scores) or reproduce subsets of the full analyses.
We expect that with advances of computing resources and infrastructure [e.g., @hayashi_brainlifeio_2024] such barriers will become progressively less limiting, but presently remain a barrier.

[^hpc_access]: While a desktop computer with 64GB of RAM and 16 cores (AMD Ryzen 9 5950X) was sufficient to fit the most resource-demanding encoding model, the process was time-consuming: fitting 15 repetitions of model fitting on one dataset size (out of total 13) for one participant required 5 hours.

**Other aspects of the software engineering and scientific computing landscape** We focused on what we believe are software practices that directly mitigate barriers to reproducibility in a research lab setting.
In doing so, we did not discuss other aspects of reproducible research such as data management [@tenopir_data_2020], workflow management [@wilkinson_applying_2025], licensing [@morin_quick_2012], software metadata [@chue_hong_fair_2022], containerization [@moreau_containers_2023; @alser_packaging_2024], or cloud-computing [@berriman_application_2013].
We think these are important and exciting avenues for reproducible research and refer interested readers to cited references for further information about these aspects.

**Impacts of AI-assisted software development** A common deterrent in adoption of software engineering practices in research is that these are time-consuming.
Recent developments in generative artificial intelligence (AI), computing systems that generate code from prompts given in natural language, are reshaping how software engineers and researchers produce and evaluate code.
How will AI-assisted programming affect reproducible research?

At the time of this writing, the landscape is still evolving, confined to individual experimentation and prototyping.
In our work, the use of AI-assisted programming was left to individual setup of each author.
Our limited experience confirmed that AI-assisted programming reduced friction in specific tasks, such as writing documentation.
While there are good reasons to expect that judicious use of AI-assisted programming will facilitate aspects of reproducible work that are currently considered time-consuming [@dupre_future_2024], using unchecked outputs of an over-confident code generation system can lead to erroneous code [e.g., @krishna_importing_2025].
Given the broad technical, ethical, and legal challenges when it comes to the use of generative AI in science [e.g., @bommasani_opportunities_2022; @birhane_science_2023; @binz_how_2025; @choudhury_promise_2025; @charness_next_2025; @shumailov_ai_2024], it will require dedicated efforts of professional communities to establish sound practices in AI-assisted development for reproducible research.

## Moving forward

We argue that software engineering practices must become a standard component in research to deliver on the promises of reproducible science.
Because the scope of engineering needs in research can vary greatly across disciplines and projects, there is no one-size-fits-all approach in terms of what practices should be adopted and when [@connolly_software_2023].
Below we briefly outline three ways of moving towards adopting software engineering practices for reproducibility.

**Starting in research teams and laboratories** Research groups and laboratories occupy an organizational level which allows swifter implementation of new policies than at the larger organizational levels with typically longer processes (e.g., university departments, schools etc.) and can be tailored to the needs of the group.
Research teams could adopt a policy that encourages reproducible science, for example that research code (and other research artifacts) be reviewed by peers before papers are submitted [e.g., @barba_reproducibility_2012].
A more comprehensive policy are internal reproducibility audits [@national_academies_of_sciences_engineering_and_medicine_reproducibility_2019], where research artifacts are validated independently by a peer within the team prior to publication.
Regardless of the comprehensiveness level, collaborative code review should be viewed as an opportunity for knowledge exchange (coding strategies, new tools, etc.) among team members and strengthening the engineering expertise of the team as opposed to solely an error finding process [@vable_code_2021].

**Towards integrated research objects and executable science** To bring reproducible computational research to the fore, the research community should move beyond regarding research code and data as accompanying outputs to static reports.
Instead, reports, code, data and other artifacts should be embedded as integrated research objects [@dupre_beyond_2022] and considered an equally important and credit-worthy output.
All components in integrated research objects would undergo peer review process, which would provide the ability to reproduce key analyses and figures with minimal efforts increasing the confidence in the reliability of the results.
An exciting avenue to realize this vision are executable research documents that combine text, data, code and visualization.
Although the idea has a venerable history [e.g., @claerbout_electronic_1992], authoring executable computational reports became more accessible recently through dedicated technical authoring and publishing frameworks such as Quarto[^quarto] and MyST Markdown that we adopted in current work.
While numerous challenges remain, for example how to archive nested computational objects and the costs of computing infrastructure [@dupre_beyond_2022], the field of computational neuroscience and neuroimaging offers promising initiatives such as the Neurolibre[^neurolibre] preprint server integrating data, code, and runtime environment [@karakuzu_neurolibre_2022].

[^quarto]: https://quarto.org/
[^neurolibre]: https://neurolibre.org/

**Professionalization of research software engineering in science** Given the central role of programming and computational techniques in science today, research software engineering should be considered as a professional discipline with its own set of standards, curricula, recognition, and scholarly communities.
Currently, code development in research does not yet enjoy equal professional standards and support as other scholarly activities and outputs, for example, teaching, scientific writing and citation practices [see @gewaltig_current_2014; @richard_mcelreath_science_2020].
Professionalization is a slowly unfolding, incremental generational process requiring change at multiple levels, such as development training opportunities [@barba_path_2024], the possibility of a career path and professional communities [@brett_research_2017; @us_research_software_engineer_association_research_2023], and incentive structures [@nosek_strategy_2019; @barba_path_2024], among others.
As a starting point, team leads should encourage interested researchers to engage with professional communities, such as the US Research Software Engineers association in the United States [@us_research_software_engineer_association_research_2023], and open source program offices (OSPOs), if one exists, at their institutions.

[^usrse]: https://us-rse.org/

# Conclusion

Trustworthy, reliable, and effective scientific progress requires science to be reproducible.
Here we reported on our effort to reproduce and replicate models of brain activity in a published fMRI story listening dataset.
Despite the results being reproducible with original code, independent replication on the basis of report alone proved challenging and required substantial effort.
We argue that to deliver on the promises of open and reproducible science, software engineering practices are essential in mitigating barriers to computational reproducibility and replicability that remain even if data and code are openly shared.
