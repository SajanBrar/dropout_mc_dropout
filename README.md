# Dropout vs. Monte Carlo Dropout: Uncertainty Estimation

A Foundations of Machine Learning project studying whether **Monte Carlo (MC) Dropout** can make neural-network predictions more informative about uncertainty, particularly when a model encounters data that differs from its training distribution.

The repository contains two complementary experiment tracks:

1. A Jupyter notebook with controlled synthetic regression and classification experiments, Gaussian Dropout, and exploratory CIFAR-10 CNN experiments.
2. A standalone CIFAR-10 training pipeline that compares a no-dropout baseline with several dropout probabilities, evaluates MC Dropout, tests out-of-distribution (OOD) behavior on SVHN, and exports data for an interactive Streamlit dashboard.

## Project objective

Neural networks can produce high-confidence predictions even when those predictions are incorrect. Dropout is commonly used as a regularizer during training; this project also investigates its use as a practical source of predictive uncertainty.

The central comparison is:

- **No Dropout:** a deterministic baseline.
- **Standard Dropout:** dropout is active during training and disabled during ordinary inference. Inference uses one deterministic forward pass.
- **MC Dropout:** dropout remains active at inference. Multiple stochastic forward passes are performed, and the resulting predictive probabilities are averaged. The predictive entropy of this average is used as an uncertainty measure.

For a classification prediction with averaged class probabilities \(\bar p_c\), predictive entropy is

\[
H(\bar p) = -\sum_c \bar p_c \log(\bar p_c).
\]

Higher entropy indicates a less concentrated predictive distribution. It is an uncertainty signal, not a guarantee that a prediction is wrong or that an input is out of distribution.

## Experiments

### 1. Synthetic-data notebook

The notebook `notebooks/Dropout and MC Dropout on Synthetic Data.ipynb` develops the comparison on smaller, interpretable datasets before moving to image classification.

**Synthetic regression**

- The underlying function is \(y_{\mathrm{true}}(x)=\sin(1.5x)\), sampled at 300 points over \([-3,3]\).
- Gaussian noise is heteroscedastic: its standard deviation grows with the magnitude of the input, using the scale factor `noise_scale * (0.1 + 0.5 * x**2)`.
- A feed-forward network with two 64-unit hidden layers is trained with mean squared error.
- No-dropout, standard-dropout, and MC-dropout variants are compared. MC predictions are estimated from repeated stochastic forward passes, and the notebook plots the predictive mean with a one-standard-deviation band.

**Synthetic classification**

- A three-class, two-dimensional dataset is generated with `make_blobs` (1,000 samples).
- A feed-forward network with hidden layers of 128 and 64 units is trained with cross-entropy loss.
- The notebook compares accuracy and calibration, computes predictive entropy, and visualizes uncertainty over the input plane.
- Expected Calibration Error (ECE) is calculated with 15 bins.

**Additional notebook experiments**

- Gaussian Dropout is explored on the synthetic regression task.
- Further CIFAR-10 CNN experiments use VGG-style architectures and compare deterministic and MC-dropout evaluation.

The notebook uses 200 stochastic forward passes for its main synthetic MC-prediction visualizations. Its CIFAR-10 exploration is separate from the 100-epoch pipeline described below.

### 2. CIFAR-10 and SVHN pipeline

The main training script is `main.py`. It trains a compact CNN on CIFAR-10 and compares one no-dropout baseline against dropout models with

\[
p \in \{0.2,\ 0.3,\ 0.4,\ 0.5\}.
\]

The CNN consists of two convolutional layers (32 channels each), ReLU activations, a 2×2 max-pooling layer, a 512-unit fully connected layer, and a 10-class output layer. In the dropout variant, dropout is applied after the 512-unit hidden layer.

The default training configuration in `main.py` is:

| Setting | Value |
|---|---|
| Training dataset | CIFAR-10 |
| OOD evaluation dataset | SVHN test split |
| Epochs | 100 per model |
| Models | 1 no-dropout baseline + 4 dropout configurations |
| Dropout probabilities | 0.2, 0.3, 0.4, 0.5 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Training batch size | 128 |
| Test batch size | 256 |
| MC passes during per-epoch accuracy evaluation | 5 |
| MC passes for dashboard inference | 30 |

CIFAR-10 is treated as the in-distribution (ID) dataset and SVHN as the OOD dataset. The script attempts to download SVHN; if that download fails, it continues without the OOD part of the experiment.

### Evaluation and dashboard

The pipeline records class predictions, ground-truth labels, confidence, correctness, and predictive entropy. The dashboard provides:

- ID-versus-OOD uncertainty histograms for CIFAR-10 and SVHN.
- Confidence-calibration/reliability diagrams.
- Inspection of incorrect predictions, including high-confidence (low-entropy) failures.
- Comparisons across dropout probabilities and inference methods.
- Training and test loss/accuracy curves.

The Streamlit application is implemented in `dashboard_viewer.py` and reads the generated files from `pickled/mc_dropout_epoch100/`.

## Reported observations

The accompanying project report describes the following results from the CIFAR-10 experiments:

- At dropout probability `p = 0.2`, the reported deterministic accuracy was **80.20%**, compared with **79.96%** for MC Dropout.
- Across the tested dropout probabilities, the report states that the accuracy difference between Standard and MC Dropout did not exceed **0.3 percentage points**.
- The report records an increase in average predictive entropy from **0.49** at `p = 0.2` to **0.72** at `p = 0.5`.
- The report describes a smaller calibration gap for MC Dropout (at most **1.8%**) than for the standard model (greater than **5%**), together with qualitative evidence that MC Dropout assigns greater uncertainty to many incorrect predictions.

These are the results documented in the project report, not guaranteed outcomes for every run. Training randomness, library versions, and hardware can change the measured values.

## Repository layout

Use the following layout when organizing the files for GitHub. The filenames below are the clean filenames expected by the commands in this README.

```text
.
├── README.md
├── requirements.txt
├── main.py
├── dashboard_viewer.py
├── pre_flight_check.py
├── test_updated_model.py
├── test_training_flow.py
├── load_and_plot_results.py
├── fix_pickle.py
├── notebooks/
│   └── Dropout and MC Dropout on Synthetic Data.ipynb
└── report/
    └── FML Project Report.pdf
```

The uploaded script files may have suffixes such as `(1)`, `(3)`, or `(4)` in their filenames. Rename them to the clean names shown above before running the commands below. In particular, `pre_flight_check.py` imports `main`, so the training script must be named `main.py`; it also reads `main.py` when checking the training configuration.

## Setup

Python 3 and `pip` are required. Create and activate a virtual environment, then install the dependencies used by the training script and dashboard:

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The notebook uses additional packages that are not all listed in `requirements.txt`. To run it, install Jupyter and the notebook-specific dependencies:

```bash
pip install jupyter scikit-learn matplotlib seaborn torchmetrics tabulate
jupyter notebook
```

Open `notebooks/Dropout and MC Dropout on Synthetic Data.ipynb` and run the cells in order. The notebook downloads CIFAR-10 when it reaches the image experiments.

## Run the CIFAR-10 experiments

From the repository root:

```bash
python test_updated_model.py
python pre_flight_check.py
python main.py
```

`test_updated_model.py` checks the model's output dimensions, dropout behavior in training and evaluation modes, and parameter count. `pre_flight_check.py` checks the model configuration, output directory, disk space, and expected training artifacts. Note that the pre-flight script imports `main.py`; because dataset initialization is at module scope in `main.py`, this import can also trigger CIFAR-10/SVHN downloads.

The default `main.py` configuration has grid search enabled and trains five models for 100 epochs each. This is a compute-intensive experiment. The script downloads CIFAR-10 and attempts to download SVHN into `./data/` if they are not already present.

## Launch the dashboard

After `main.py` has completed successfully and generated the pickle files, run:

```bash
streamlit run dashboard_viewer.py
```

The dashboard expects these files:

```text
pickled/mc_dropout_epoch100/
├── dashboard_data.pkl
└── grid_search_results.pkl
```

The training script also saves model checkpoints (`.pth`) in this directory. The dashboard will report missing data if training has not generated the required files.

## Generated artifacts

The main pipeline writes its outputs to `pickled/mc_dropout_epoch100/`, including:

- `cifar10_no_dropout_final.pth`
- `cifar10_dropout_p_0.2.pth` through `cifar10_dropout_p_0.5.pth`
- `grid_search_results.pkl`
- `dashboard_data.pkl`

The datasets are stored under `data/`. These files are generated locally and are not required to understand the source code. They can be regenerated by running the experiments.

## Notes on auxiliary scripts

- `test_updated_model.py`: standalone checks for the current CNN architecture and dropout behavior.
- `pre_flight_check.py`: checks the expected training setup before the long run.
- `fix_pickle.py`: converts tensors nested in the dashboard pickle into CPU NumPy arrays. It overwrites `dashboard_data.pkl`; make a backup before running it.
- `load_and_plot_results.py`: an older plotting helper. It expects `pickled/mc_dropout/training_metrics.pkl`, which is not produced by the current `main.py` pipeline, and it needs `matplotlib`.
- `test_training_flow.py`: a development-time serialization/data-structure test that contains an earlier model definition. Treat it as a historical helper rather than as an integration test for the current `main.py` implementation.

## Implementation notes and limitations

- In `main.py`, device selection currently uses Apple Metal (MPS) when available and otherwise falls back to CPU. It does **not** currently select a CUDA GPU. The notebook has its own CUDA-or-CPU device selection.
- The current default path is the grid-search path (`ENABLE_GRID_SEARCH = True`). The alternate single-model branch in `main.py` contains an older dropout-constructor call; do not disable grid search without first updating that branch to use the current `CIFAR10ModelWithDropout(p=...)` interface.
- The OOD analysis depends on SVHN being downloaded successfully.
- Predictive entropy is one uncertainty measure. It should be interpreted alongside accuracy, calibration, and the data context rather than as a standalone detector.

## References

The project report discusses the following foundations:

- Srivastava et al. (2014), *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*.
- Gal and Ghahramani (2016), *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*.

## Contributors

Jayasree M, Sajanjit Singh, and Shreyas Shetty  
CS725 — Foundations of Machine Learning  
Project report dated November 26, 2025.

## License

No license is specified in the supplied project files. Add a `LICENSE` file before permitting reuse or redistribution of this repository.
