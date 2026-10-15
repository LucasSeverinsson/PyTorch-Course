# Week 01 — Tensors and Numerical Computing

**Course:** PyTorch from First Principles  
**Phase:** Foundations  
**Suggested effort:** 6–8 hours  
**Prerequisites:** Python, basic NumPy, elementary linear algebra  
**Status:** Not yet assessed

> **How to use this chapter:** Read each lesson, predict outputs *before* running code, and implement the exercises yourself. Write explanations, derivations, solutions, mistakes, and new vocabulary in `answers.md`. This file deliberately contains **no exercise solutions**. Use your Week 1 chat for hints, debugging, and review.

## 0. Learning goals and checklist

By the end of the week, you should be able to:

- [+] Explain scalar, vector, matrix, tensor, axis, shape, rank/order, dtype, and device.
- [ ] Predict the shapes resulting from indexing, transposes, reshapes, reductions, and broadcasting.
- [ ] Explain and use `*`, `@`, `sum`, `mean`, and `keepdim`.
- [ ] Distinguish a view, a copy, and a non-contiguous tensor.
- [ ] Compute per-feature means, variances, standardized data, and sample covariance.
- [ ] Derive and implement pairwise Euclidean distances using two vectorized methods.
- [ ] Check results against NumPy and `torch.cdist`, and compare a loop with a vectorized implementation.

**Scope:** We are not training neural networks yet. Automatic differentiation starts in Week 2.

## 1. Environment and workspace

On macOS, from the VS Code terminal:

```bash
mkdir -p pytorch-from-scratch/{notebooks,exercises/week01,projects/linear_algebra_toolkit,tests}
cd pytorch-from-scratch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy matplotlib scipy scikit-learn jupyter ipykernel pytest
```

If the repository already exists, simply activate its virtual environment and install any missing packages; don't recreate it. Create `notebooks/week01_tensors.ipynb`, or use an ordinary `.py` file if preferred. Select the `.venv` interpreter/kernel.

```python
import torch
import numpy as np

print('torch:', torch.__version__)
print('NumPy:', np.__version__)
print('MPS available:', torch.backends.mps.is_available())
print('CUDA available:', torch.cuda.is_available())

x = torch.randn(3, 4)
print(x.shape, x.dtype, x.device)
```

**Important:** MPS is only available on supported Apple hardware/software. All Week 1 exercises work on CPU; prefer CPU until you are comfortable with shape and dtype semantics.

## 2. Lesson A — What is a tensor?

In this course, a **tensor** is a multidimensional array. A scalar has shape `()`, a vector of length four has shape `(4,)`, a 2 × 3 matrix has shape `(2, 3)`, and a batch of 16 RGB images with height 32 and width 32 might have shape `(16, 3, 32, 32)`.

A tensor's **rank** (also called *number of dimensions* or *order* here) is the number of axes, **not** the rank of a matrix in linear algebra. Its **shape** gives the size of each axis. Its **number of elements** is the product of the axis sizes.

For supervised learning, `X.shape == (n, p)` often means `n` observations and `p` features. If `W.shape == (p, h)`, then `X @ W` has shape `(n, h)`.

```python
scalar = torch.tensor(5.0)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])

for t in (scalar, vector, matrix):
    print('shape:', t.shape, 'ndim:', t.ndim,
          'numel:', t.numel(), 'dtype:', t.dtype, 'device:', t.device)
```

PyTorch closely resembles NumPy for numerical array operations. Two major additional capabilities matter for later weeks: *automatic differentiation* and execution on supported accelerators such as CUDA GPUs and Apple MPS.

### Exercise A — Predict before executing

Write predictions in `answers.md` first. Then verify with code.

1. What are the shapes and ranks of `torch.tensor(3)`, `torch.arange(7)`, `torch.zeros(2, 1, 4)`?
2. For a tensor with shape `(5, 8, 3)`, how many elements does it have? What does axis 1 refer to?
3. Make the matrix `[[1,2,3],[4,5,6],[7,8,9]]`. Extract its second row and third column using indexing/slicing. Describe their shapes.
4. Predict the outputs and shapes of `X[0]`, `X[:, 0]`, `X[1:3, :2]`, and `X[1, 2]` for `X = torch.arange(12).reshape(3,4)`.
5. Create arrays representing 100 observations with 12 features each, and a batch of 64 grayscale 28 × 28 images. State and justify your shape convention.

## 3. Lesson B — Operations and reductions

**Elementwise multiplication** combines matching (or broadcast-compatible) entries: `(A * B)[i,j] = A[i,j] B[i,j]`. **Matrix multiplication** sums products across a shared dimension: `(A @ B)[i,k] = Σ_j A[i,j] B[j,k]`.

For `A` with shape `(m,n)` and `B` with shape `(n,p)`, `A @ B` has shape `(m,p)`. For reductions, `dim=0` reduces across rows, while `dim=1` reduces across columns in a 2D matrix. Setting `keepdim=True` retains the reduced axis with size one, often useful for later broadcasting.

```python
A = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
B = torch.ones(3, 4)

print('A @ B:', (A @ B).shape)
print('column means:', A.mean(dim=0))
print('row means:', A.mean(dim=1))
print('row means, keepdim:', A.mean(dim=1, keepdim=True).shape)
print('transpose:', A.T.shape)
```

### Exercise B — Shape reasoning and basic linear algebra

1. Predict the shapes of `(2,3) @ (3,5)`, `(4,2,3) @ (3,6)`, and `(2,3) * (2,3)`. Why is `(2,3) @ (2,3)` invalid?
2. Using your 3 × 3 matrix from Exercise A, calculate each row's mean and each column's sum. No Python loops.
3. With `A = torch.randn(6, 4)`, predict the shapes of `A.sum(dim=0)`, `A.sum(dim=1)`, and `A.sum(dim=1, keepdim=True)`.
4. Verify the matrix-multiplication shape rule using random tensors and an assertion.
5. Explain why `A * B` and `A @ B` generally produce different results even when both happen to be valid.

## 4. Lesson C — Broadcasting

PyTorch compares tensor shapes **from the last dimension toward the first**. Each pair of aligned dimensions is compatible if their sizes are equal, one of them is 1, or one aligned dimension is absent. In most cases, size-1 dimensions are logically repeated; PyTorch usually avoids physically copying all the repeated values.

Example: `X.shape == (32,10)` and `b.shape == (10,)`. Alignment from the right gives `(32,10)` and `(1,10)`, resulting in shape `(32,10)`. This is useful for adding a feature-wise bias to every observation.

```python
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
b = torch.tensor([1., 10., 100., 1000.])
print((X + b).shape)

col = X.mean(dim=0, keepdim=True)
row = X.mean(dim=1, keepdim=True)
print('col:', col.shape, 'row:', row.shape)
```

**Common trap:** `(n,)` is not identical to `(n,1)`. A row/column mismatch can silently generate a large matrix rather than the desired vector.

### Exercise C — Broadcast predictions

For each pair, write **valid/invalid**, and the output shape if valid. Explain the trailing-dimension alignment.

| Left | Right |
|---|---|
| `(32,10)` | `(10,)` |
| `(5,1,7)` | `(1,3,1)` |
| `(4,3)` | `(4,)` |
| `(8,1)` | `(8,)` |
| `(2,3,4)` | `(3,4)` |
| `(2,3)` | `(3,2)` |

Then implement the following without loops:

1. Subtract each column mean from a matrix of shape `(100, 5)`.
2. Subtract each row mean from that matrix.
3. Explain which dimension requires `keepdim=True` in each case and why.
4. Construct a case where broadcasting succeeds but gives a mathematically unintended result. Describe a shape assertion that would catch it.

## 5. Lesson D — Reshape, transpose, views, and copies

`reshape` changes the logical shape without changing element count. `view` also changes shape but requires compatible memory layout/strides. `transpose(dim0,dim1)` swaps two axes, while `permute(...)` reorders all specified axes. Both typically return **views** (shared storage) with modified strides rather than moving data immediately.

A tensor may be **non-contiguous** after a transpose. `reshape` may return a view if possible or allocate a copy if necessary. Do not assume that modifying the result of every reshape will modify the original.

```python
x = torch.arange(12).reshape(3, 4)
y = x.transpose(0, 1)
print('shapes:', x.shape, y.shape)
print('contiguous:', x.is_contiguous(), y.is_contiguous())
print('strides:', x.stride(), y.stride())
print('flattened:', y.reshape(-1).shape)
```

### Exercise D — Memory and layout

1. Predict the shapes after: `x = torch.randn(2,3,4)`, `x.permute(2,0,1)`, `x.transpose(0,2)`, and `x.reshape(6,4)`.
2. On the tensor above, explain why `x.permute(2,0,1)` does not mean the same operation as `x.reshape(4,2,3)`.
3. Try `view(-1)` on a transposed 2D tensor and record what happens; compare with `reshape(-1)`.
4. Investigate whether `y = x[:, :2]` shares storage with `x`. What happens if you modify one entry of `y`? Perform this experiment on a separate disposable tensor.
5. Explain when shared storage is useful and when it can cause difficult bugs.

## 6. Lesson E — Dtypes, devices, and numerical precision

`float32` (approximately 7 significant decimal digits) is common in ML because it uses less memory and can be faster on accelerators. `float64` (approximately 16 significant decimal digits) is often useful for sensitive numerical calculations and numerical gradient checks. Precision, memory, and accelerator support affect which dtype makes sense. Integer tensors behave differently from floating-point tensors for operations such as means and division.

A tensor lives on a **device**, typically CPU, CUDA, or MPS. All tensors participating in an operation ordinarily need compatible devices, and many operations require matching or appropriately promoted dtypes. Different devices may differ in supported operations and numerical behavior.

```python
x32 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
x64 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
print(x32.dtype, x64.dtype)
print('sum32:', x32.sum().item())
print('sum64:', x64.sum().item())
```

### Exercise E — Numerical experiments

1. Estimate memory usage for a `(10000, 1000)` tensor in `float32` and `float64` (assume 4 and 8 bytes per element respectively).
2. Compare `float32` and `float64` when summing many small values. Report what you observe without assuming one small experiment proves universal behavior.
3. Identify your machine's available PyTorch devices. Keep your Week 1 calculations on CPU unless you want an optional device experiment.
4. Explain why mixing CPU and MPS/CUDA tensors in a computation is problematic.

## 7. Lesson F — Statistics using tensors

Let `X` be an `n × p` feature matrix: rows are observations, columns are variables. The feature mean vector is

$$\bar{x}_j = \frac{1}{n}\sum_{i=1}^{n} X_{ij}.$$

The sample feature variance is

$$s_j^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_{ij}-\bar{x}_j)^2,$$

and the standardized observations are

$$Z_{ij} = \frac{X_{ij}-\bar{x}_j}{s_j}.$$

The sample covariance matrix is

$$S = \frac{1}{n-1}(X-\mathbf{1}\bar{x}^{\mathsf T})^{\mathsf T}(X-\mathbf{1}\bar{x}^{\mathsf T}),$$

with shape `(p,p)`. **Convention matters:** `correction=1` gives the sample variance denominator `n-1`; `correction=0` uses denominator `n`. If a feature is constant, its standard deviation is zero and naive standardization divides by zero. Choose and document a policy for these columns.

Starter data only (derive your implementation yourself):

```python
torch.manual_seed(42)
X = torch.randn(100, 4, dtype=torch.float64)
X[:, 1] = 2.0 * X[:, 0] + 0.3 * torch.randn(100, dtype=torch.float64)
```

### Exercise F — Statistical functions

1. Implement `feature_mean(X)`, `sample_variance(X)`, `standardize(X)`, and `sample_covariance(X)` using elementary PyTorch tensor operations. Do **not** use built-in `torch.cov` as your initial implementation.
2. For a non-constant dataset, verify that each standardized column has approximately zero mean and sample variance one.
3. Verify your sample covariance is symmetric, has shape `(p,p)`, and matches `np.cov(X.numpy(), rowvar=False)` within a reasonable numerical tolerance.
4. Compare your variance result with `X.var(dim=0, correction=1)`. How does `correction=0` change the answer?
5. Add input checks for `X.ndim == 2`, at least two observations, and floating-point dtype. Define a policy for zero-variance features and test it.

## 8. Lesson G — Pairwise Euclidean distances and vectorization

For two collections of row vectors, `X ∈ R^{n×d}` and `Y ∈ R^{m×d}`, the distance matrix `D ∈ R^{n×m}` has elements

$$D_{ij} = \lVert X_{i,:}-Y_{j,:}\rVert_2.$$

**Method 1: broadcasting.** Consider the shapes of `X[:, None, :]` and `Y[None, :, :]`. Their difference has shape `(n,m,d)`; reducing the last axis produces pairwise distances.

**Method 2: squared norms and matrix multiplication.** Expand

$$\lVert x-y\rVert_2^2 = \lVert x\rVert_2^2 + \lVert y\rVert_2^2 - 2x^{\mathsf T}y.$$

This permits construction of all squared pairwise distances with rowwise squared norms and `X @ Y.T`, without explicitly constructing an `(n,m,d)` difference tensor. Floating-point cancellation may produce slightly negative *computed* squared distances; clamp to a minimum of zero before taking square roots. The matrix method can also lose accuracy when very close vectors have very large norms.

For verification, PyTorch provides `torch.cdist(X,Y)`. The methods have different intermediate-memory profiles: broadcasting materializes an `(n,m,d)` array, whereas the Gram-matrix method mainly needs an `(n,m)` result (plus inputs).

### Exercise G — Main algorithm exercise

1. Derive the squared-distance identity by expanding the inner product `(x-y)ᵀ(x-y)` in your `answers.md`.
2. Implement `pairwise_broadcast(X,Y)` **without explicit Python loops**. Write down the shape after each intermediate operation.
3. Implement `pairwise_matmul(X,Y)` using rowwise squared norms and matrix multiplication; clamp computed squared distances at zero before square-rooting.
4. Check both against `torch.cdist` with `torch.testing.assert_close` on several small random inputs, as well as identical points, repeated points, and vectors with large magnitude.
5. Explain differences in speed, intermediate memory, and numerical stability. Check that `D.shape == (n,m)`, `D >= 0`, and that all entries are finite for your tested finite inputs.
6. Optional: implement a simple loop-based version solely as a reference, then time all three approaches.

**Do not assume** all random inputs will show the same performance ordering. Benchmark fairly and report tensor sizes, dtype, device, number of runs, and whether accelerator synchronization is needed (if using an accelerator).

## 9. Mini-project — `linear_algebra_toolkit/`

Create a small, reusable Python package or module, rather than keeping every function in your notebook.

Suggested layout:

```text
pytorch-from-scratch/
├── answers.md                       # Your notes and solutions, not part of the lesson
├── notebooks/
│   └── week01_tensors.ipynb
├── exercises/
│   └── week01/
├── projects/
│   └── linear_algebra_toolkit/
│       ├── __init__.py
│       ├── operations.py            # Your implementations
│       └── benchmark.py             # Your timing experiment
└── tests/
    └── test_week01.py
```

Implement (signatures are suggestions; you own the code):

```python
def feature_mean(X): ...
def sample_variance(X): ...
def standardize(X): ...
def sample_covariance(X): ...
def pairwise_broadcast(X, Y): ...
def pairwise_matmul(X, Y): ...
```

**Requirements:**

- Validate shapes and document expected input/output dimensions.
- Handle or clearly reject non-floating inputs and invalid `n`.
- Choose and document a policy for zero-variance features.
- Include tests comparing statistical functions with NumPy and distances with `torch.cdist`.
- Test small cases where you can check the result by hand.
- Include a loop versus vectorization timing experiment; do not present raw runtimes without input sizes and conditions.
- Write a short README or notes section explaining numerical and memory tradeoffs.

**Optional stretch:** Evaluate whether the matrix-identity distance computation is robust when `X` and `Y` contain nearly identical large vectors; compare `float32` and `float64`.

## 10. Theoretical self-assessment (write answers in `answers.md`)

Do not look up an answer until you have tried to reason it through.

**Q1.** What distinguishes a scalar, vector, matrix, and higher-rank tensor? Define shape, rank, and axis.  
**Q2.** For `A.shape == (32,10)` and `b.shape == (10,)`, what is the shape of `A+b`, and why?  
**Q3.** What are the broadcasting compatibility rules? Give an invalid example and explain the mismatch.  
**Q4.** How do `*` and `@` differ? State dimension constraints.  
**Q5.** What does `keepdim=True` do and why can it help?  
**Q6.** Distinguish `view`, `reshape`, `transpose`, and `permute`. When may reshaping create a copy?  
**Q7.** What issues arise from `float32` vs `float64` and from mixing CPU/GPU/MPS tensors?  
**Q8.** Derive the squared Euclidean-distance identity. Why is it useful, and why might clamping be necessary?  
**Q9 (extension).** When is the `(n,m,d)` broadcasting implementation less memory-efficient than the matrix identity?  
**Q10 (extension).** Why does `X.T @ X / (n-1)` fail to give the sample covariance matrix when the feature columns are not centered?

## 11. Suggested work sessions

| Session | Focus | Evidence to save in `answers.md` |
|---|---|---|
| 1 | Lessons A–B | Shape predictions, first examples, Q1 and Q4 |
| 2 | Lessons C–D | Broadcasting table, stride/view observations, Q2–Q3, Q5–Q6 |
| 3 | Lessons E–F | Precision notes, statistical derivations, numerical tests, Q7 |
| 4 | Lesson G | Distance derivation and both vectorized implementations, Q8 |
| 5 | Mini-project | Unit tests, timing results, reflection, Q9–Q10 |

The sessions are recommendations, not a calendar commitment.

## 12. Completion criteria and reflection

Before requesting a Week 1 review, make sure you can:

- Predict shapes **before running code**, including broadcasts that fail.
- Derive and implement the two distance algorithms without copying reference code.
- Explain why each statistical reduction is along a particular axis.
- Compare your implementations with NumPy and `torch.cdist`, including meaningful edge cases.
- Explain at least one numerical and one memory/performance tradeoff.

In `answers.md`, write a short reflection: **What made sense? What was confusing? Which mistakes did you correct? What should be repeated?** Ask your Week 1 chat to review your answers and code. The week is complete only after you confirm completion or demonstrate these competencies; the chat should not assume it.

## 13. References

- Official PyTorch tensor tutorial: https://docs.pytorch.org/tutorials/beginner/basics/tensor_tutorial.html
- PyTorch broadcasting semantics: https://docs.pytorch.org/docs/stable/notes/broadcasting.html
- `torch.cdist`: https://docs.pytorch.org/docs/stable/generated/torch.cdist.html
- `torch.Tensor.view`: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
- `torch.reshape`: https://docs.pytorch.org/docs/stable/generated/torch.reshape.html
- `torch.var`: https://docs.pytorch.org/docs/stable/generated/torch.var.html
- PyTorch numerical accuracy notes: https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html

---

**Next week:** Autograd and computational graphs. Resist the temptation to use built-in training APIs for Week 1: the point is to build and verify the numerical foundations first.
