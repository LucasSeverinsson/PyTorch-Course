# PyTorch from First Principles — Project Context and Course Syllabus

> **Purpose:** This document is the persistent learning plan and teaching context for the **PyTorch course** ChatGPT project. It is independent of the Home Credit Default Risk project. Use it to guide lessons, exercises, reviews, assessments, and progress discussions.

## 1. Learner profile and goals

- Background: Engineering Physics, mathematical statistics, probability, optimization, time-series analysis, and Python/NumPy/scikit-learn.
- Goal: Develop **deep practical and mathematical understanding** of PyTorch and neural-network training, not merely use high-level APIs.
- Preference: **Control and transparency.** Implement ideas with tensors and derivations first, then introduce `torch.nn`, `torch.optim`, and other abstractions.
- Longer-term motivation: Be prepared for research-oriented ML and potential radar/sensor-signal applications at Acconeer AB. Do not presume that the eventual thesis topic is known.
- Keep this course **separate** from the credit-default project. Use synthetic data, standard vision datasets, and signal-processing datasets instead.

## 2. Teaching instructions for future conversations

1. Teach one concept at a time: mathematical intuition → equations → small worked example → code → exercises.
2. Ask the learner to attempt derivations and coding problems independently. Do not provide the full solution immediately unless asked; start with hints and diagnostics.
3. Explain tensor **shapes**, data types, gradient flow, and why each line of code is needed.
4. Distinguish a pedagogical from-scratch implementation from efficient production PyTorch.
5. Encourage verification: assertions, numerical gradient checking, unit tests, baselines, plots, and reproducibility.
6. When reviewing answers to theoretical questions, assess correctness, point out missing assumptions, and ask a targeted follow-up where necessary.
7. Use the question bank below as **self-assessment**, not as a requirement to answer everything at once. Mix conceptual, mathematical, and experimental reasoning.
8. Avoid assuming a module is completed unless the learner says they completed it or demonstrates its competencies. Track progress explicitly in conversation rather than guessing.
9. Recommended cadence: **12 weeks at about 6–8 hours/week**, adjustable. Depth and competence matter more than dates.
10. Prefer official PyTorch documentation for API behavior, and check current documentation if version-sensitive.

## 3. Learning outcomes

By the end, the learner should be able to:

- Manipulate tensors confidently, including broadcasting and matrix operations.
- Derive and verify gradients, understand reverse-mode automatic differentiation, and debug gradient flow.
- Implement and compare optimizers and training algorithms.
- Build networks with basic tensor operations and with `nn.Module`.
- Create reusable datasets, training loops, evaluation routines, and checkpoints.
- Train and investigate CNNs and temporal models.
- Design custom objectives including probabilistic losses.
- Run fair, reproducible experiments and analyze robustness and uncertainty.
- Implement simple custom differentiable operations.
- Complete a research-style capstone using noisy complex-valued sensor signals.

## 4. Study sequence at a glance

| Phase | Week | Module | Main deliverable |
|---|---:|---|---|
| Foundations | 1 | Tensors and numerical computing | Linear algebra toolkit |
| Foundations | 2 | Autograd and computational graphs | Gradient checker |
| Foundations | 3 | Optimization from scratch | Optimizer comparison |
| Neural networks | 4 | Backpropagation and MLPs | MLP without `torch.nn` |
| Neural networks | 5 | `nn.Module` and training loops | FashionMNIST MLP |
| Neural networks | 6 | Datasets and experimental workflow | Reusable training engine |
| Architectures | 7 | CNNs | Image CNN comparison |
| Architectures | 8 | Time-series models | Synthetic signal classifier |
| Architectures | 9 | Custom losses and uncertainty | Heteroscedastic regression |
| Research | 10 | Evaluation and robustness | Robustness/ablation report |
| Research | 11 | Advanced autograd | Differentiable custom component |
| Research | 12 | Radar-like sensor capstone | Research-style mini-project |

---

# Phase I — Foundations

## Week 1 — Tensors and mathematical operations

**Objectives:** Understand tensor dimensionality, shapes, indexing, broadcasting, reductions, dtype/device, views versus copies, vectorization, and how PyTorch compares with NumPy.

**Theory:** Tensor notation; matrix multiplication; broadcasting rules; Euclidean distances; covariance/standardization; numerical precision and memory.

**Coding exercises:**
1. Generate matrices of different shapes; predict results of reshaping, transposition, reductions, and broadcasting **before** running code.
2. Compute per-feature mean, variance, standardization, and covariance in PyTorch; check against NumPy.
3. Calculate pairwise Euclidean distances without Python loops (first with broadcasting, then using squared norms and matrix multiplication); check against `torch.cdist`.
4. Compare a loop-based implementation with a vectorized one; describe performance differences.

**Mini-project:** `linear_algebra_toolkit/` with functions, shape checks, unit tests, and timing comparisons.

**Theoretical questions:**
1. What distinguishes a scalar, vector, matrix, and higher-rank tensor? What do **shape**, **rank**, and **axis** mean?
2. For `A.shape == (32, 10)` and `b.shape == (10,)`, what is the shape of `A + b`, and why is it valid?
3. Under what conditions can two tensors be broadcast together? Give an example that fails.
4. Explain the difference between elementwise multiplication (`*`) and matrix multiplication (`@`). State the dimension conditions for each.
5. What does `keepdim=True` do in a reduction, and why can it matter for later broadcasting?
6. What is the practical difference between `view`, `reshape`, `transpose`, and `permute`? When can a reshape produce a copy?
7. What problems can arise when mixing `float32` and `float64`, or CPU and GPU/MPS tensors?
8. Derive the identity `||x-y||² = ||x||² + ||y||² - 2 xᵀy`. Why is it useful for pairwise distances, and what numerical precaution may be needed?

**Completion criteria:** Can predict tensor shapes, write vectorized computations, and verify against reference implementations.

**Reference:** https://docs.pytorch.org/tutorials/beginner/basics/tensor_tutorial.html

## Week 2 — Automatic differentiation and computational graphs

**Objectives:** Understand computational graphs, chain rule, reverse-mode autodiff, leaf tensors, gradient accumulation, `requires_grad`, `backward`, `autograd.grad`, and gradient checking.

**Coding exercises:**
1. Derive and check derivatives of `f(x,y) = x²y + sin(xy)`.
2. Call `backward()` twice under an appropriate graph setup and observe gradient accumulation; explain how to reset gradients.
3. Compare autodiff gradients with centered finite differences for scalar functions of vectors.
4. Compute a Jacobian of a small vector-valued function; compare shapes with your mathematical expectation.

**Mini-project:** `gradient_checker/`, with tolerances, explanatory failure cases, and tests.

**Theoretical questions:**
1. What is a computational graph, and how does the chain rule operate on it?
2. Why is reverse-mode automatic differentiation efficient for a scalar loss depending on many parameters?
3. Explain the difference between symbolic differentiation, numerical finite differences, and automatic differentiation.
4. What is a leaf tensor? When does PyTorch populate `.grad` by default?
5. Why do gradients accumulate instead of automatically resetting after `backward()`?
6. What is the difference between `loss.backward()` and `torch.autograd.grad()`?
7. Why does an optimizer update usually occur under `torch.no_grad()`? How is that different from `detach()`?
8. Why can finite-difference checks disagree with autograd near nondifferentiable points or with inappropriate step sizes?
9. For a vector output and scalar input, what does `backward()` require, and how is a vector–Jacobian product involved?

**Completion criteria:** Can derive and verify gradients and explain autograd graph behavior.

**Reference:** https://docs.pytorch.org/tutorials/beginner/basics/autograd_tutorial.html

## Week 3 — Optimization from scratch

**Objectives:** Understand empirical risk minimization, batch/mini-batch/stochastic gradient descent, convergence, momentum, Adam, learning rates, regularization, and stability.

**Coding exercises:**
1. Fit `y = Xw + b + ε` with manual gradient descent; compare with ordinary least squares where the solution is well-defined.
2. Implement mini-batch SGD and momentum using tensor updates.
3. Implement Adam from the update equations and compare its parameter trajectory against `torch.optim.Adam`.
4. Vary feature scaling, initialization, batch size, and learning rate; graph the effects on convergence.

**Mini-project:** `optimizer_comparison/`, including plots and short analysis.

**Theoretical questions:**
1. Define an empirical risk objective. How does it relate to maximum likelihood for Gaussian observation noise?
2. Derive the gradient of mean squared error for linear regression in matrix notation.
3. Distinguish full-batch, mini-batch, and stochastic gradient descent; how do their gradient estimates differ?
4. Why can a learning rate be too small or too large? Explain using a quadratic objective.
5. How does momentum change an update compared with vanilla gradient descent?
6. Why does Adam maintain estimates of both first and second gradient moments? What does bias correction do?
7. How does feature scaling affect conditioning and optimizer convergence?
8. Compare L2 regularization and decoupled weight decay. Are they always equivalent?
9. What can convergence plots reveal, and what can they *not* prove about generalization?

**Completion criteria:** Can implement several optimizers and interpret their behavior.

**Reference:** https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

---

# Phase II — Neural networks

## Week 4 — Neural networks and backpropagation

**Objectives:** Understand affine layers, activations, cross-entropy, parameter initialization, multistage chain rule, and backpropagation through an MLP.

**Coding exercises:**
1. Create an MLP with tensors: `H = ReLU(XW1 + b1)`, `logits = HW2 + b2`.
2. Derive gradients for a small network and compare with autograd; explicitly identify tensor shapes.
3. Train a binary classifier on `make_moons` using manually defined parameters.
4. Test `tanh` and ReLU; inspect loss curves and gradients.

**Mini-project:** `mlp_from_scratch/` (without `nn.Module` or built-in optimizers in its initial version).

**Theoretical questions:**
1. Why do stacked linear layers without nonlinear activations collapse into a single affine transformation?
2. What does a hidden layer learn, and how do hidden width/depth affect representational capacity?
3. Derive the derivative of ReLU where it is differentiable. What happens at zero in typical implementations?
4. Write the chain-rule expression for the gradient of the loss with respect to the first weight matrix in a two-layer network.
5. What is the difference between logits and probabilities? Why is `BCEWithLogitsLoss` usually preferable to applying sigmoid and then BCE separately?
6. What are vanishing and exploding gradients, and how can activations and initialization affect them?
7. Why should biases often have different initializations from weights?
8. How would you diagnose a network whose training loss stays nearly constant?

**Completion criteria:** Can explain and implement backpropagation through an MLP.

**Reference:** https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html

## Week 5 — `nn.Module` and training loops

**Objectives:** Organize models as `nn.Module`, understand `nn.Parameter`, model modes, losses and optimizers, and write an explicit training/validation loop.

**Coding exercises:**
1. Convert the Week 4 model into a subclass of `nn.Module`.
2. Write `train_one_epoch` and `evaluate` functions with correct `model.train()`/`model.eval()` and gradient modes.
3. Train an MLP on FashionMNIST with mini-batches; track validation loss and accuracy.
4. Compare SGD and Adam, and save/load a `state_dict` checkpoint.

**Mini-project:** `fashion_mnist_mlp/`.

**Theoretical questions:**
1. What does `nn.Module` provide beyond a normal Python class?
2. What is an `nn.Parameter`, and how are trainable parameters registered?
3. What is the difference between `forward()` and explicitly invoking `model(x)`?
4. Explain the correct order of `zero_grad`, forward pass, loss, `backward`, and optimizer update. Why?
5. Why do `model.train()` and `model.eval()` matter for dropout and batch normalization?
6. Why should gradients generally be disabled during validation? Is `model.eval()` sufficient to do that?
7. Distinguish `state_dict` checkpoints from serializing an entire Python model object.
8. What does the training–validation loss gap tell you, and what alternative explanations are possible?

**Completion criteria:** Can implement and debug training/evaluation loops independently.

**Reference:** https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

## Week 6 — Data pipelines and experimental design

**Objectives:** Build reusable `Dataset`/`DataLoader` pipelines, careful data splits, preprocessing, reproducibility, tests, experiment logging and checkpointing.

**Coding exercises:**
1. Build a custom `Dataset` and connect it to `DataLoader`.
2. Refactor code into `datasets.py`, `models.py`, `train.py`, `evaluate.py` (or an equivalent package).
3. Make runs reproducible to a practical level; record seed, configuration, metrics, and versions.
4. Implement checkpoint resume and verify loaded weights reproduce evaluation outputs under the same environment.

**Mini-project:** `reusable_training_engine/`.

**Theoretical questions:**
1. What is the division of responsibility between `Dataset` and `DataLoader`?
2. Why do training, validation, and test sets have different roles? How can repeated test-set use invalidate conclusions?
3. What constitutes data leakage? Give two examples involving preprocessing or related observations.
4. When should training examples be shuffled, and when is naive shuffling inappropriate?
5. What are batch size, `num_workers`, and `pin_memory`, and how might they affect performance?
6. Why doesn't setting a random seed always guarantee bitwise reproducibility across hardware and library versions?
7. What information should a checkpoint contain if the goal is to resume training faithfully?
8. Why is it useful to separate experiment configuration from model and training implementation?

**Completion criteria:** Can train the same model from a reproducible, modular script.

**Reference:** https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

---

# Phase III — Architectures and probabilistic models

## Week 7 — Convolutional neural networks

**Objectives:** Understand kernels, PyTorch-style cross-correlation, stride, padding, pooling, receptive fields, parameter sharing, and CNN inductive bias.

**Coding exercises:**
1. Implement a small 2D cross-correlation operation from tensor operations and compare with `nn.Conv2d`.
2. Compute convolution output shapes by hand and verify with PyTorch.
3. Train a CNN on FashionMNIST (or CIFAR-10 if resources allow).
4. Compare against a similarly budgeted MLP; evaluate parameter counts, learning curves, and errors.

**Mini-project:** `image_cnn_comparison/`.

**Theoretical questions:**
1. How does a CNN differ from a fully connected network in connectivity and parameter sharing?
2. Explain mathematical convolution versus cross-correlation. Which convention is used in `nn.Conv2d`?
3. Derive the output height/width of a convolution from input size, kernel, stride, padding, and dilation.
4. What is a receptive field, and how does it grow across stacked convolutions?
5. Why can translation-related inductive biases help image learning? When might they be unhelpful?
6. What is the purpose of pooling, and what information can it discard?
7. How can a CNN overfit even though it shares parameters?
8. How would you separate the effect of architecture from the effect of parameter count or training budget?

**Completion criteria:** Can implement and reason about CNN architecture choices.

**Reference:** https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

## Week 8 — Time-series models and 1D CNNs

**Objectives:** Understand sequential data preparation, windows, 1D convolutions, temporal receptive fields, GRU/LSTM basics, forecasting/classification distinctions, and leakage.

**Coding exercises:**
1. Simulate sine waves and noisy/nonstationary signals with known labels.
2. Create a custom sequence dataset with documented tensor shape conventions.
3. Train a 1D CNN and a GRU on signal classification.
4. Test performance on frequencies, amplitudes, and noise regimes absent from training.

**Mini-project:** `synthetic_signal_classifier/`.

**Theoretical questions:**
1. How does time-series classification differ from forecasting and sequence-to-sequence prediction?
2. Explain the typical `Conv1d` input layout `(batch, channels, length)` and how it differs from common recurrent-model layouts.
3. What determines the temporal receptive field of a stack of 1D convolutions?
4. How does a recurrent network summarize a sequence? What problem do GRU/LSTM gates address?
5. Why are neighboring sliding windows often highly dependent? How should that influence evaluation splits?
6. What types of frequency or phase patterns might a CNN learn from a waveform?
7. Why might a Fourier-based baseline outperform a neural network on a small, clean periodic-signal problem?
8. How would you design a test for robustness to an unseen sampling frequency or changed signal amplitude?

**Completion criteria:** Can build sequential-data pipelines and compare temporal architectures fairly.

**Reference:** https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

## Week 9 — Custom losses, regularization, and predictive uncertainty

**Objectives:** Understand loss design from likelihoods, heteroscedastic Gaussian regression, output constraints, regularization, dropout, gradient clipping and calibration.

**Coding exercises:**
1. Generate regression data where observation variance depends on the input.
2. Train a neural network outputting mean and log-variance with a custom Gaussian negative log-likelihood.
3. Compare the result with mean-squared-error regression and inspect prediction intervals.
4. Evaluate the empirical coverage of nominal prediction intervals on held-out data.

**Mini-project:** `probabilistic_regression/`.

**Theory note:** For `Y|X=x ~ Normal(mu(x), sigma(x)^2)`, negative log likelihood per observation, excluding constants, is `log(sigma) + (y-mu)^2 / (2 sigma^2)`. Parameterize `log_var` or a transformed scale to ensure positive variance and manage numerical stability.

**Theoretical questions:**
1. How is a standard loss function derived from a probabilistic model and maximum likelihood?
2. Derive the Gaussian negative log-likelihood for a model predicting `mu(x)` and `sigma²(x)`.
3. Why must predicted variance be constrained to be positive? Name two parameterization strategies.
4. Distinguish aleatoric uncertainty from epistemic uncertainty. Which does heteroscedastic Gaussian regression primarily address?
5. How does dropout differ between training and evaluation?
6. Compare L2 regularization, weight decay, early stopping, and data augmentation.
7. What does empirical interval coverage measure? Can good average coverage conceal poor conditional coverage?
8. Why might an NLL-trained model predict implausibly large or small variances, and how would you diagnose the issue?

**Completion criteria:** Can derive, implement, and evaluate a custom probabilistic loss.

**Reference:** https://docs.pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html

---

# Phase IV — Research skills and capstone

## Week 10 — Evaluation, robustness, and generalization

**Objectives:** Plan controlled experiments, avoid leakage, use baselines, evaluate distribution shift, analyze variability across seeds, calibration, and ablations.

**Coding exercises:**
1. Evaluate the Week 8 models under several held-out noise/amplitude regimes.
2. Perform an ablation (e.g., remove preprocessing or reduce sequence length).
3. Repeat training with multiple random seeds; report mean/spread with uncertainty appropriate to the experimental design.
4. Write a short report explaining failures, limitations, and what evidence supports each conclusion.

**Mini-project:** `robustness_report/`.

**Theoretical questions:**
1. What does out-of-distribution generalization mean, and how does it differ from IID test performance?
2. What is an ablation study? How should it be designed to support a causal interpretation about a model component?
3. Why should all models be compared under reasonably matched data splits, metrics, and tuning budgets?
4. How do randomness from training initialization and randomness from dataset sampling differ?
5. What does a bootstrap confidence interval estimate? Under what dependence assumptions might naive resampling be invalid?
6. What is calibration, and why is it distinct from accuracy or discrimination?
7. When are accuracy, F1, AUROC, or a task-specific cost more appropriate?
8. Give three ways a benchmark comparison can appear impressive while failing to establish a useful improvement.

**Completion criteria:** Can write a defensible, reproducible evaluation report.

**Reference:** https://docs.pytorch.org/tutorials/

## Week 11 — Advanced autograd and PyTorch internals

**Objectives:** Use Jacobians/Hessians, vectorization and `torch.func`, custom autograd functions, numerical validation, performance profiling and differentiable physical components.

**Coding exercises:**
1. Compute a Jacobian and Hessian on a small nonlinear mathematical model; check expected dimensions and symmetry where appropriate.
2. Implement Newton's method on a small well-conditioned optimization problem, with damping or a safeguard as needed.
3. Implement a custom `torch.autograd.Function` with a hand-written backward rule and validate using `gradcheck` in double precision.
4. Integrate a differentiable mathematical transform into a small trainable model.

**Mini-project:** `custom_differentiable_component/`.

**Theoretical questions:**
1. What is a Jacobian? What is a Hessian? What do their dimensions tell you?
2. What is a vector–Jacobian product, and why is it central to reverse-mode autodiff?
3. When would higher-order derivatives be useful in optimization or ML research?
4. What are the risks of naive Newton updates when a Hessian is singular or indefinite?
5. When do you need a custom `autograd.Function`, and when is composing ordinary differentiable PyTorch operations enough?
6. What must a custom backward operation return for each forward input?
7. Why is `gradcheck` commonly run with double-precision tensors?
8. Distinguish computational graph construction, execution time, and memory usage; how can intermediate activations affect memory?

**Completion criteria:** Can construct, validate and debug a custom differentiable operation.

**References:** https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html and https://docs.pytorch.org/docs/stable/func.html

## Week 12 — Capstone: Robust learning from radar-like sensor signals

**Objectives:** Combine signal simulation, PyTorch models, statistical signal-processing baselines, reproducible experimentation, and robustness assessment.

**Proposed task:** Model a **simplified** coherent sensor observing periodic displacement:

- `d(t) = A sin(2π f t)`;
- `z(t) = exp(j 4π d(t)/λ) + n(t)` for complex noise `n(t)`.

This is a **pedagogical simulation**, not a complete physical model of an Acconeer sensor. Extend to varying frequencies, amplitudes, phase offsets, noise, drift, and multiple reflectors only when justified.

**Build:**
1. A simulation generator with controllable parameters and saved metadata.
2. Signal-processing baseline(s), e.g. phase extraction/unwrapping + FFT/periodogram for frequency estimation.
3. A neural baseline and a 1D CNN or GRU on a clearly defined representation (IQ channels, unwrapped phase, spectrogram, etc.).
4. A train/validation/test procedure that separates underlying simulation settings or recording groups as appropriate.
5. Robustness experiments for SNR, amplitude, frequency range, and distribution shift.
6. A brief report with methods, plots, uncertainty/error summaries, failure cases, limitations, and next research steps.

**Theoretical questions:**
1. Starting with a coherent round-trip path, why is displacement associated with a phase term proportional to `4π d/λ` in a simplified monostatic model?
2. Why are IQ measurements complex-valued, and what do amplitude and phase represent?
3. What is phase wrapping? Under what sampling and motion conditions can unwrapping become unreliable?
4. How do measurement duration and sample rate influence FFT frequency resolution and the Nyquist limit?
5. Why should FFT/periodogram estimation be included as a baseline before training a neural network?
6. How might multipath reflections, body movement, missing data, or sensor drift invalidate the simplified model?
7. What constitutes data leakage when many windows are generated from the same underlying simulated signal or measurement session?
8. What experimental evidence would justify claiming that a neural model improves on a classical signal-processing method?
9. How would you quantify and report estimation uncertainty across changing noise conditions?
10. Which additional steps would be required before moving from a synthetic demonstration to real Acconeer A121 data?

**Final deliverables:** source code, tests, README with commands, reproducible configurations, plots, and a short technical write-up.

**Completion criteria:** Can defend the experimental design and the mathematical model, rather than just report a single accuracy number.

**References:** https://docs.acconeer.com/ ; https://github.com/acconeer/acconeer-python-exploration

---

## 5. Suggested repository organization

```text
pytorch-from-scratch/
├── README.md
├── pyproject.toml
├── .gitignore
├── notebooks/
├── exercises/
│   ├── week01/
│   └── ...
├── src/
│   └── pytorch_course/
│       ├── models/
│       ├── datasets/
│       ├── training/
│       └── utils/
├── projects/
│   ├── mlp_from_scratch/
│   ├── image_cnn_comparison/
│   ├── synthetic_signal_classifier/
│   └── radar_capstone/
├── tests/
└── reports/
```

Use notebooks for mathematical exploration and plots; use `.py` modules for reusable logic and experiments. Add tests early. Keep the final project entirely separate from company or credit-default datasets.

## 6. Study and assessment protocol

For each module, the recommended order is:

1. **Concept check:** Answer 3–5 theoretical questions without notes.
2. **Derivation:** Complete the relevant math by hand or in Markdown/LaTeX.
3. **Coding exercise:** Implement the operation from scratch where feasible.
4. **Verification:** Use small examples, assertions, numerical checks, and reference implementations.
5. **Mini-project:** Build a complete standalone result.
6. **Reflection:** Write what worked, what failed, and which assumptions matter.

**Suggested grading rubric for theoretical answers:**

- **0 — Not yet:** major conceptual or mathematical misunderstanding.
- **1 — Developing:** partly correct but missing an essential distinction or justification.
- **2 — Competent:** mathematically and conceptually correct with reasonable assumptions.
- **3 — Strong:** correct, precise, explains limitations, and connects theory with implementation.

A learner is ready to move on when they can explain the central ideas and complete the mini-project without relying on a copied solution. The questions are diagnostic, not a timed exam.

## 7. Initial setup

```bash
mkdir pytorch-from-scratch
cd pytorch-from-scratch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch numpy matplotlib scipy scikit-learn jupyter pytest
```

```python
import torch
print(torch.__version__)
print(torch.backends.mps.is_available())  # Relevant on compatible Apple Silicon Macs
x = torch.randn(3, 4)
print(x.shape)
```

**First exercise:** Implement pairwise Euclidean distances between the rows of two matrices without loops, using broadcasting and the squared-norm identity, and verify with `torch.cdist`.

## 8. Core links

- PyTorch Basics: https://docs.pytorch.org/tutorials/beginner/basics/
- What is `torch.nn` really?: https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html
- PyTorch tutorials: https://docs.pytorch.org/tutorials/
- Dive into Deep Learning (free text): https://d2l.ai/
- Acconeer documentation: https://docs.acconeer.com/
- Acconeer Python Exploration Tool: https://github.com/acconeer/acconeer-python-exploration

## 9. Progress log (update as modules are completed)

| Week | Status | Notes / weak areas | Completed date |
|---:|---|---|---|
| 1 | Not started | | |
| 2 | Not started | | |
| 3 | Not started | | |
| 4 | Not started | | |
| 5 | Not started | | |
| 6 | Not started | | |
| 7 | Not started | | |
| 8 | Not started | | |
| 9 | Not started | | |
| 10 | Not started | | |
| 11 | Not started | | |
| 12 | Not started | | |

**End goal:** Be able to read a research paper specifying a reasonably tractable neural architecture or custom loss, reproduce its core math, implement it in PyTorch, verify gradients, and design experiments that test whether the method actually works.
