# Instructions

## Reproducing the published figures

This guide shows how to reproduce the figures from the precomputed model performance scores from all experiments (model-brain correlations) using the provided code.
It requires the installation of the relevant dependencies and the correlation results of various encoding model runs.

#### 1. Clone repository, navigate into created repository directory

```bash
git clone git@github.com:GabrielKP/enc.git
cd enc
```

#### 2. Setup a virtual environment and install dependencies

Next, create a virtual environment with Python 3.12 using your preferred manager.

For example, using conda:

```bash
# conda environment
conda create -n enc python=3.12
conda activate enc
```

Or using [uv](https://docs.astral.sh/uv/) (in project root directory):

```bash
uv venv --python 3.12 --seed # creates .venv environment folder and installs python and pip
source .venv/bin/activate # activate the environment
```

Now that you have python and pin ready, install the project code and its dependencies:
```bash
# install package in editable mode
pip install -e .

```

#### 3. Install git-annex and datalad

Install [git-annex and datalad](https://handbook.datalad.org/en/latest/intro/installation.html). This is required in order to be able to download the data from the OpenNeuro repository.

#### 4. Download the subject-specific cortical surface data

With Datalad and git-annex installed, you can use our wrapper script that downloads the data from [the original fMRI data repository](https://github.com/OpenNeuroDatasets/ds003020.git) with Datalad:

```bash
# Only download data required for plotting results
python src/encoders/download_data.py --figures [--data_dir DATA_DIR]
```

Without `--data_dir DATA_DIR` the data will be downloaded into the folder `ds003020` of the project directory.
To download the data into a custom dir, specify `--data_dir DATA_DIR` (it is recommended to call the last folder `ds003020` as that is the default dataset name).

#### 5. Setup/check `config.yaml`.

The `config.yaml` should be created automatically when you run the download script in step 4. If not, you can copy the [example config file](https://github.com/GabrielKP/enc/blob/main/config.example.yaml). Make sure that the `DATA_DIR` key points to the folder where you downloaded the data to in step 4.:

```yaml
CACHE_DIR: .cache
DATA_DIR: ds003020 # <-- should point to the download location
RUNS_DIR: runs
INKSCAPE_PATH: /path/to/inkscape
INKSCAPE_VERSION: X.Y.Z
TR_LEN: 2.0
```

> [!NOTE]
> If you get the error `ImportError: cannot import name 'getargspec' from 'inspect'` then try to update your datalad version `python -m pip install datalad --upgrade`

#### 6. Download the pre-computed experiment results (model performance scores)

Download [`runs.zip`](https://osf.io/download/g9cy3) from the OSF repository and unzip the contents such that you have a `runs` directory which contains the results of individual experiments in separate subfolders:

```
runs/extension_ridgeCV
runs/replication_ridge_huth
runs/replication_ridgeCV
runs/reproduction
```

> [!NOTE]
> The `runs` folder should be placed in the project root (i.e. at the same level as `data`, `src` etc.)

#### 7. Install [inkscape](https://inkscape.org/) (required for plotting):

Open the `config.yaml` and set the following values accordingly.

```yaml
INKSCAPE_PATH: path/to/inkscape/binary
INKSCAPE_VERSION: X.Y.Z
```

For mac, you usually can [find inkscape as described here](https://stackoverflow.com/a/22085247).

#### 8. Configure pycortex (required for plotting):

**Using the provided script**

You can use the script in the repository to configure the pycortex:

```bash
python src/encoders/update_pycortex_config.py
```

**Configure manually**

Find the location of your pycortext config with the python terminal.
Type `python` in the command line with the virtual environment activated.
Then execute following commands:

```py
import cortex
cortex.options.usercfg
```

This should give you a path to the config file, copy it and exit the terminal.

```bash
# open the file with an editor of choice (e.g. vim)
vim path/to/options.cfg
```

Modify the entry at `filestore` to `DATA_DIR/derivative/pycortex-db`.
Whereas `DATA_DIR` is the directory of the Lebel et al. data repository.
E.g. if you did not specify a custom datadir, `DATA_DIR` then it is `/path/to/this/repository/ds003020`.

