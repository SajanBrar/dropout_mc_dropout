# Dropout vs. Monte Carlo Dropout: Uncertainty Estimation

A project studying whether **Monte Carlo (MC) Dropout** can make neural-network predictions more informative about uncertainty, particularly when a model encounters data that differs from its training distribution.

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

**Synthetic regression**

- The underlying function is \(y_{\mathrm{true}}(x)=\sin(1.5x)\), sampled at 300 points over \([-3,3]\).
- Gaussian noise is heteroscedastic: its standard deviation grows with the magnitude of the input, using the scale factor `noise_scale * (0.1 + 0.5 * x**2)`.
- A feed-forward network with two 64-unit hidden layers is trained with mean squared error.
- No-dropout, standard-dropout, and MC-dropout variants are compared. MC predictions are estimated from repeated stochastic forward passes, and the notebook plots the predictive mean with a one-standard-deviation band.

**Synthetic classification**

- A three-class, two-dimensional dataset is generated with `make_blobs` (1,000 samples).
- A feed-forward network with hidden layers of 128 and 64 units is trained with cross-entropy loss.

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

CIFAR-10 is treated as the in-distribution (ID) dataset and SVHN as the OOD dataset. 

## References

- Srivastava et al. (2014), *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*.
- Gal and Ghahramani (2016), *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*.
