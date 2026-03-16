# NOTE — CNN From Scratch (CIFAR-10)

## 1) Project Intent

This project was built in two parallel directions:

- **Learning direction:** implement CNN components manually with NumPy in `src/`
- **Practical direction:** train/evaluate with PyTorch + CUDA for faster experiments and better accuracy

`src/` is the educational core. The notebook is the practical prediction entry point.

---

## 2) What Was Implemented (Learning Track)

### Core layers (NumPy)
- `Conv2D` forward/backward
- `MaxPool2D` forward/backward
- `AvgPool2D` forward/backward
- `Flatten` forward/backward
- `DenseLayer` forward/backward
- Activations and loss (`ReLU`, `Softmax`, cross-entropy)
- `SGD` optimizer support for dense + conv style gradients

### Training infra (NumPy)
- CIFAR-10 loading/preprocess
- Mini-batch training loop
- Shuffle per epoch
- Learning-rate step decay
- Weight decay support
- Checkpoint saving in scratch train script

---

## 3) Practical Track (PyTorch + CUDA)

### Why this track exists
The pure NumPy training path is CPU-bound and slow for larger experiments.
PyTorch + CUDA was used for practical performance and stronger accuracy.

### Final result highlights
- GPU detected and used correctly (RTX 3050 Laptop GPU)
- CUDA-enabled PyTorch environment confirmed
- Best training run exceeded target range
- **Best reported test accuracy: 91.58%**

---

## 4) Current Inference Flow

Use notebook:
- `notebooks/predict_cifar10.ipynb`

It now loads:
- `model/cifar10_cnn.pt` (PyTorch checkpoint)

Important:
- this checkpoint is PyTorch format (`model_state`, `optimizer_state`, ...)
- it is not interchangeable with scratch NumPy checkpoint keys (`conv_w`, `dense_w`, ...)

---

## 5) Repository Publishing Notes

### Keep in repository
- `src/` (learning implementation)
- `notebooks/predict_cifar10.ipynb` (prediction/demo)
- `model/cifar10_cnn.pt` (if size policy allows)
- this note and README

### Optional cleanup before push
- remove temporary checkpoints not used
- ensure notebook output is either intentionally kept or cleared
- add dataset/checkpoint download instructions if large files are excluded

---

## 📝 Session Log

### Session 1 — Project Kickoff

**What we discussed:**
- Established project rules: learn-by-doing, no AI-generated code without permission
- Created the full roadmap (above)
- Libraries allowed: `numpy`, `scipy`, `matplotlib`, data loading utilities

---

## Phase 1.1 — Understanding CIFAR-10

### 📖 Theory

**CIFAR-10** (Canadian Institute For Advanced Research, 10 classes) is one of the most widely used benchmark datasets in computer vision.

#### What's inside?

| Property | Value |
|---|---|
| Total images | **60,000** |
| Training set | **50,000** images |
| Test set | **10,000** images |
| Image size | **32 × 32 pixels** |
| Color channels | **3** (Red, Green, Blue) |
| Number of classes | **10** |

#### The 10 Classes

| Label | Class Name |
|-------|-----------|
| 0 | Airplane ✈️ |
| 1 | Automobile 🚗 |
| 2 | Bird 🐦 |
| 3 | Cat 🐱 |
| 4 | Deer 🦌 |
| 5 | Dog 🐕 |
| 6 | Frog 🐸 |
| 7 | Horse 🐴 |
| 8 | Ship 🚢 |
| 9 | Truck 🚛 |

Each class has exactly **6,000 images** (5,000 training + 1,000 testing), so the dataset is **perfectly balanced**.

#### Image Representation (Key Concept!)

Each image is a **3D array (tensor)** with shape:

```
(Height, Width, Channels) = (32, 32, 3)
```

- **Height** = 32 pixels (rows)
- **Width** = 32 pixels (columns)
- **Channels** = 3 (R, G, B color channels)

Each pixel value is an integer from **0 to 255**:
- `0` = no intensity (black for that channel)
- `255` = full intensity (brightest for that channel)

So a single image is essentially **32 × 32 × 3 = 3,072 numbers**.

The entire training set can be thought of as a 4D array:
```
(N, H, W, C) = (50000, 32, 32, 3)
```
Where `N` is the number of images. (Some formats store it as `(N, C, H, W)` — we'll discuss this later.)

#### Why CIFAR-10?

- **Small images (32×32)** → fast to train, perfect for learning from scratch without GPU
- **Real-world objects** → more challenging than simple digits (like MNIST)
- **Balanced classes** → no class imbalance to worry about
- **Well-studied** → easy to compare your results with benchmarks

---

## Phase 1.2 — Data Preprocessing (Theory)

### 📖 Theory

#### 1. Normalization

Pixel values are `0–255` (integers). Neural networks train better with small float values.

**Simplest approach:** divide by 255.0 → values become `0.0 to 1.0`

**Why?** Large inputs → large gradients → unstable training. Normalization keeps everything in a manageable range.

#### 2. One-Hot Encoding

Labels are integers (e.g., `6` = frog). For Cross-Entropy loss, we need **one-hot vectors**:

```
Label 6  →  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
Label 0  →  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

A vector of length 10 with `1` at the class position, `0` everywhere else.

**How to build it with NumPy:**
- Create a zero matrix with `np.zeros((num_samples, num_classes))`
- Use array indexing to put `1` in the right position

---

## Phase 2.1 — Neurons & Linear Transformation

### 📖 Theory

#### What is a Neuron?

A neuron is the smallest unit of a neural network. It takes **inputs**, multiplies each by a **weight**, adds them up, adds a **bias**, and produces an **output**:

```
y = w₁x₁ + w₂x₂ + w₃x₃ + ... + wₙxₙ + b
```

In compact form: **y = W·x + b** (dot product + bias)

#### Single Neuron Example

Imagine a neuron with 3 inputs:

```
Inputs:   x = [2.0, 3.0, 1.0]
Weights:  w = [0.5, -1.0, 2.0]
Bias:     b = 0.1

y = (2.0×0.5) + (3.0×-1.0) + (1.0×2.0) + 0.1
y = 1.0 + (-3.0) + 2.0 + 0.1
y = 0.1
```

In NumPy: `y = np.dot(w, x) + b`

#### From One Neuron to a Layer

A **layer** is just multiple neurons working in parallel. If we have:
- **n** inputs (e.g., 3072 for a flattened CIFAR-10 image)
- **m** neurons in the layer (e.g., 128)

Then:
- **W** is a matrix of shape `(n, m)` — each column is one neuron's weights
- **b** is a vector of shape `(m,)` — one bias per neuron
- **x** is a vector of shape `(n,)` — the input

```
y = x · W + b
```

Result `y` has shape `(m,)` — one output per neuron.

#### With a Batch of Inputs

We don't feed one image at a time — we use **batches** (e.g., 32 images at once):

- **X** has shape `(batch_size, n)` — e.g., `(32, 3072)`
- **W** has shape `(n, m)` — e.g., `(3072, 128)`
- **b** has shape `(m,)` — e.g., `(128,)`

```
Y = X @ W + b     # @ is matrix multiplication in NumPy
```

Result **Y** has shape `(batch_size, m)` — e.g., `(32, 128)`

#### Weight Initialization (Why It Matters)

Weights should NOT start at zero (all neurons would learn the same thing!). Common approaches:
- **Random small values**: `np.random.randn(n, m) * 0.01`
- Biases can start at zero: `np.zeros(m)`

---

## Phase 2.2 — Activation Functions

### 📖 Theory

#### Why do we need Activation Functions?

Without activation functions, a neural network is just stacking linear transformations:

```
Layer 1: y₁ = X @ W₁ + b₁
Layer 2: y₂ = y₁ @ W₂ + b₂
```

But stacking linear functions gives... another linear function! `y₂ = X @ (W₁ @ W₂) + (b₁ @ W₂ + b₂)` — this is equivalent to a **single layer**. No matter how many layers you stack, it's still linear.

Activation functions add **non-linearity**, letting the network learn complex patterns (curves, edges, shapes).

#### 1. ReLU (Rectified Linear Unit) — Most popular for hidden layers

```
ReLU(x) = max(0, x)
```

- If input is **positive** → keep it
- If input is **negative** → output 0

Example:
```
Input:  [-2.0, 0.5, -0.1, 3.0, -1.0]
Output: [ 0.0, 0.5,  0.0, 3.0,  0.0]
```

**Why ReLU?** Simple, fast, and avoids the "vanishing gradient" problem. It's the default choice for hidden layers.

**NumPy:** `np.maximum(0, x)`

#### 2. Sigmoid — Maps values to (0, 1)

```
Sigmoid(x) = 1 / (1 + e^(-x))
```

- Very negative x → output ≈ 0
- x = 0 → output = 0.5
- Very positive x → output ≈ 1

Useful for **binary classification** (yes/no), but less used in modern deep networks because of vanishing gradients.

**NumPy:** `1 / (1 + np.exp(-x))`

#### 3. Softmax — Used at the LAST layer for classification

```
Softmax(xᵢ) = e^(xᵢ) / Σ e^(xⱼ)    for all j
```

Takes a vector of raw scores and converts to **probabilities that sum to 1**.

Example:
```
Input (raw scores):  [2.0, 1.0, 0.5]
Output (probabilities): [0.59, 0.24, 0.17]   ← sums to 1.0!
```

The highest score gets the highest probability. We pick the class with the highest probability as our prediction.

**NumPy:** `np.exp(x) / np.sum(np.exp(x))` (but needs a trick for numerical stability — subtract max first!)

#### Summary: When to use what?

| Activation | Where | Purpose |
|-----------|-------|---------|
| **ReLU** | Hidden layers | Adds non-linearity, fast |
| **Sigmoid** | Rarely used now | Maps to (0,1), binary tasks |
| **Softmax** | Last layer only | Converts scores to probabilities |

---

## Phase 2.3 — Loss Functions (Cross-Entropy)

### 📖 Theory

#### What is a Loss Function?

A loss function measures **how wrong** the network's prediction is. It outputs a single number:
- **High loss** = bad prediction (far from correct answer)
- **Low loss** = good prediction (close to correct answer)

The goal of training is to **minimize the loss**.

#### Cross-Entropy Loss — The Standard for Classification

For classification tasks (like CIFAR-10), we use **Categorical Cross-Entropy Loss**.

**Formula (for one sample):**

```
Loss = -Σ yᵢ · log(pᵢ)     for all i (classes)
```

Where:
- `yᵢ` = true label (one-hot encoded): `[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]` (class 6)
- `pᵢ` = predicted probability (from Softmax): `[0.01, 0.02, 0.05, 0.03, 0.02, 0.01, 0.80, 0.03, 0.02, 0.01]`

Since `y` is one-hot, most terms are `0 × log(...)  = 0`. Only the **true class** survives:

```
Loss = -log(p_true_class)
```

#### Example — Intuition

**Good prediction** (true class = 6, model says 80% for class 6):
```
Loss = -log(0.80) = 0.22   ← low loss ✅
```

**Bad prediction** (true class = 6, model says 5% for class 6):
```
Loss = -log(0.05) = 3.00   ← high loss ❌
```

**Perfect prediction** (true class = 6, model says 100%):
```
Loss = -log(1.0) = 0.0     ← zero loss 🎯
```

Notice: `-log(x)` is **large when x is small** (bad prediction) and **small when x is large** (good prediction).

#### For a Batch of Samples

Average the loss across all samples in the batch:

```
Loss_batch = -(1/N) × Σ Σ yᵢⱼ · log(pᵢⱼ)
```

Where N = number of samples, i = sample index, j = class index.

#### Numerical Stability

Just like Softmax, `log(0)` = negative infinity 💥. So we add a tiny number:

```
Loss = -log(p + 1e-8)     # 1e-8 = 0.00000001
```

This prevents `log(0)` crashes while barely changing the result.

---

## Phase 2.4 — Backpropagation (Theory)

### 📖 Theory

#### The Big Picture

Training a neural network has 3 steps repeated over and over:

```
1. FORWARD PASS  → feed input through network → get prediction
2. COMPUTE LOSS  → compare prediction to true label → get error
3. BACKWARD PASS → compute gradients → update weights to reduce error
```

Step 3 is **backpropagation** — the most important algorithm in deep learning.

#### What are Gradients?

A **gradient** tells you: *"If I increase this weight a tiny bit, how much does the loss change?"*

```
gradient of loss w.r.t. weight w = ∂Loss/∂w
```

- If gradient is **positive** → increasing w makes loss **worse** → decrease w
- If gradient is **negative** → increasing w makes loss **better** → increase w
- If gradient is **zero** → w is at a good spot

#### The Chain Rule — Heart of Backpropagation

Consider a simple network:

```
Input x → [Linear: z = wx + b] → [ReLU: a = max(0,z)] → [Loss: L]
```

We want `∂L/∂w` (how does changing w affect the loss?). By the **chain rule**:

```
∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w
```

We break it into small, easy pieces and multiply them together!

Each piece is simple:
- `∂z/∂w = x` (derivative of `wx + b` w.r.t. `w` is just `x`)
- `∂a/∂z = 1 if z > 0, else 0` (derivative of ReLU)
- `∂L/∂a` = comes from the loss function

#### How Backprop Works — Step by Step

We compute gradients **backwards** (from loss → output → hidden → input):

```
Step 1: Compute ∂L/∂a  (gradient from loss)
Step 2: Compute ∂L/∂z = ∂L/∂a × ∂a/∂z  (pass through ReLU)
Step 3: Compute ∂L/∂w = ∂L/∂z × ∂z/∂w  (get weight gradient)
Step 4: Compute ∂L/∂b = ∂L/∂z × ∂z/∂b  (get bias gradient)
```

#### Key Derivatives You'll Need

| Layer/Function | Forward | Backward (derivative) |
|---|---|---|
| Linear: `z = Wx + b` | `z = x @ W + b` | `∂L/∂W = xᵀ @ ∂L/∂z` , `∂L/∂b = sum(∂L/∂z)` , `∂L/∂x = ∂L/∂z @ Wᵀ` |
| ReLU: `a = max(0, z)` | `a = np.maximum(0, z)` | `∂L/∂z = ∂L/∂a × (z > 0)` |
| Softmax + CE Loss | combined | `∂L/∂z = predictions - targets` ← beautifully simple! |

The last one is amazing: the gradient of **Softmax + Cross-Entropy combined** is just `predicted - actual`. This is why they're always used together!

#### Weight Update (Gradient Descent)

Once we have gradients, update the weights:

```
w_new = w_old - learning_rate × ∂L/∂w
```

- **learning_rate** (e.g., 0.01) controls how big each step is
- Too large → overshoots, unstable
- Too small → learns too slowly

#### Visual Summary

```
FORWARD:  x ──→ [W₁,b₁] ──→ ReLU ──→ [W₂,b₂] ──→ Softmax ──→ Loss
                                                                  │
BACKWARD: ∂L/∂W₁ ←── ∂L/∂a₁ ←── ∂L/∂z₂ ←── ∂L/∂a₂ ←── ∂L/∂p ←┘
          (update!)                           (update!)
```

Gradients flow **backwards** through the network — hence "back-propagation"!

---

## Phase 2.5 — Dense (Fully Connected) Layer Implementation

### 📖 Theory

A Dense layer is a **class** with two main methods:

#### `forward(input)` — Already learned in Phase 2.1!
```
output = input @ W + b
```
But we must **save the input** because we need it during backward pass.

#### `backward(grad_output)` — Using Phase 2.4!

`grad_output` is `∂L/∂output` — the gradient flowing back from the next layer.

We compute 3 things:
1. **Gradient for weights**: `∂L/∂W = inputᵀ @ grad_output`
2. **Gradient for biases**: `∂L/∂b = sum(grad_output, axis=0)`
3. **Gradient for input** (to pass to previous layer): `∂L/∂input = grad_output @ Wᵀ`

#### Class Structure

```
class DenseLayer:
    __init__(self, input_size, output_size):
        # Initialize W with small random values
        # Initialize b with zeros
        # Initialize grad_W and grad_b to None

    forward(self, input):
        # Save input for backward pass
        # Return input @ W + b

    backward(self, grad_output):
        # Compute self.grad_W = ?
        # Compute self.grad_b = ?
        # Return gradient for input (to pass backward)
```

#### Why save the input?

Look at `∂L/∂W = inputᵀ @ grad_output` — we need the input from the forward pass! So during `forward()`, we store `self.input = input`.

#### Shape Reference

| Variable | Shape | Example |
|---|---|---|
| `input` (X) | `(batch_size, input_size)` | `(32, 3072)` |
| `W` | `(input_size, output_size)` | `(3072, 128)` |
| `b` | `(output_size,)` | `(128,)` |
| `output` | `(batch_size, output_size)` | `(32, 128)` |
| `grad_output` | `(batch_size, output_size)` | `(32, 128)` |
| `grad_W` | `(input_size, output_size)` | `(3072, 128)` |
| `grad_b` | `(output_size,)` | `(128,)` |
| `grad_input` | `(batch_size, input_size)` | `(32, 3072)` |

---

## Phase 2.6 — Building a Simple MLP

### 📖 Theory

An **MLP (Multi-Layer Perceptron)** stacks Dense layers with activations:

```
Input (3072) → Dense(128) → ReLU → Dense(10) → Softmax → Prediction
```

For CIFAR-10:
- Input: flattened image `(32×32×3 = 3072)`
- Hidden layer: 128 neurons with ReLU
- Output layer: 10 neurons (one per class) with Softmax

#### The Full Training Loop

```
for each epoch:
    for each mini-batch:
        1. FORWARD: pass batch through all layers
        2. LOSS: compute cross-entropy
        3. BACKWARD: compute gradients through all layers (reversed)
        4. UPDATE: adjust all weights using gradients
```

#### Forward Pass (data flows →)
```python
# Flatten: (32, 32, 32, 3) → (32, 3072)
z1 = dense1.forward(x_batch)      # (32, 3072) → (32, 128)
a1 = relu(z1)                      # (32, 128) → (32, 128)
z2 = dense2.forward(a1)            # (32, 128) → (32, 10)
predictions = softmax(z2)          # (32, 10) → (32, 10) probabilities
loss = cross_entropy_loss(predictions, y_batch)
```

#### Backward Pass (gradients flow ←)
```python
# Softmax + CE combined gradient
grad = predictions - y_batch        # (32, 10) — beautifully simple!
grad = dense2.backward(grad)        # (32, 10) → (32, 128)
grad = grad * (z1 > 0)              # ReLU backward: zero out negatives
grad = dense1.backward(grad)        # (32, 128) → (32, 3072)
```

#### Weight Update
```python
learning_rate = 0.01
dense1.w -= learning_rate * dense1.grad_W
dense1.b -= learning_rate * dense1.grad_b
dense2.w -= learning_rate * dense2.grad_W
dense2.b -= learning_rate * dense2.grad_b
```

#### Softmax for Batches

Your current `softmax` works for 1D input. For batches (2D), you need to apply it **per row**:
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```
`axis=1, keepdims=True` ensures each row is normalized independently.

---

#### 🚨 Crucial Real-World Training Details

1. **Zero-Mean Centering (Avoid "Dying ReLUs"):** Raw image pixels are `[0.0, 1.0]`. If fed like this, dense layers output huge negative numbers causing ReLUs to output 0 forever. *Fix:* Scale inputs to `[-1.0, 1.0]` by doing `(x / 255.0 - 0.5) / 0.5`.
2. **Gradient Scaling:** Since our loss is averaged over the batch `(batch_size)`, the gradient must also be divided by `batch_size`. Otherwise, weight updates are massive and unstable. *Fix:* `grad = (predictions - y_batch) / batch_size`.
3. **Data Shuffling:** We must shuffle the dataset between epochs using `np.random.permutation()`, or the network will memorize the batch order instead of learning patterns.
4. **He Initialization:** In `DenseLayer.__init__`, scale the initial random weights by `np.sqrt(2.0 / input_size)` instead of `0.01`. This specifically prevents vanishing/exploding gradients when using ReLU!

---

## Phase 3 — Convolutional Neural Network Layers

Why did our MLP only get 51% accuracy? Because when we flattened the `32x32` image into a `3072` vector, we **destroyed all spatial information**. A pixel at the top left and a pixel at the bottom right became next-door neighbors in the 1D array.

A CNN solves this by keeping the image 2D (or 3D with color channels) and looking at **small local regions** at a time.

---

### Phase 3.1 — Convolution Operation (Theory)

#### 📖 Theory

#### 1. What is a Convolution?

Instead of weights connecting to *every* input pixel (like a Dense layer), a Convolutional Layer has a small grid of weights called a **Kernel** (or Filter) — typically `3x3` or `5x5`.

This Kernel **slides** across the input image. At each position, it does an element-wise multiplication with the pixels it's currently covering, and sums the result to produce a single output number.

*Analogy:* Imagine shining a `3x3` flashlight across a dark painting. The flashlight represents the kernel looking for a specific pattern (like a vertical edge or a color gradient).

#### 2. The Math (Sliding Window)

Let's say we have a `5x5` grayscale input image, and a `3x3` kernel:

1. Place the `3x3` kernel at the top-left corner of the image.
2. Multiply the 9 image pixels by the 9 kernel weights.
3. Sum those 9 numbers together.
4. Add the bias parameter.
5. Store the result in the top-left of the output grid.
6. Slide the kernel right by 1 pixel and repeat.

#### 3. Hyperparameters: Stride and Padding

**Stride (`s`):** How many pixels the kernel shifts at each step.
- Stride 1: Slides over by 1 pixel (standard, overlapping).
- Stride 2: Slides over by 2 pixels (skips pixels, reduces output size by half).

**Padding (`p`):** Adding a border of zeros around the input image.
- *Why?* Two reasons:
  1. The kernel can't go off the edge, so normally the output shrinks. Padding prevents shrinking ("Valid" vs "Same" padding).
  2. It ensures the pixels on the very edge of the image get looked at more than once.

#### 4. The Output Shape Formula 🧠

If you have an input of size `W_in` (width), a kernel of size `K`, padding `P`, and stride `S`:

```
W_out = (W_in - K + 2*P) / S + 1
```
*(Same formula applies to Height).*

*Example:* Input is `32x32`, Kernel is `3x3`, Padding is `1`, Stride is `1`.
`W_out = (32 - 3 + 2(1)) / 1 + 1 = 31 / 1 + 1 = 32`.
(The padding kept the output size exactly the same!)

#### 5. Channels & Multiple Filters

- **Depth of Input:** A color image has 3 channels (RGB). So a `3x3` kernel is actually a `3x3x3` block of weights! It looks at all 3 colors simultaneously.
- **Multiple Filters:** One filter might look for horizontal lines, another for red blobs. If we use 16 different filters, our output will have 16 channels (called "Feature Maps").

---

## Phase 3.2 — Convolution Layer Forward Pass (Theory, No Code)

### 📖 Goal

Implement a `Conv2D` forward pass from scratch using NumPy loops first (clear and correct), then optimize later if needed.

### 1) Tensor Shapes (Recommended Convention)

- Input batch `X`: `(N, C_in, H, W)`
- Filters `W`: `(C_out, C_in, K_h, K_w)`
- Bias `b`: `(C_out,)`
- Output `Y`: `(N, C_out, H_out, W_out)`

Where:

$$
H_{out} = \left\lfloor \frac{H + 2P - K_h}{S} \right\rfloor + 1,
\quad
W_{out} = \left\lfloor \frac{W + 2P - K_w}{S} \right\rfloor + 1
$$

### 2) Forward Computation Logic

For each sample `n`, each output filter `f`, and each output position `(i, j)`:

1. Map to input top-left:
    - `h_start = i * stride`
    - `w_start = j * stride`
2. Extract input patch of shape `(C_in, K_h, K_w)`
3. Elementwise multiply with filter `W[f]`
4. Sum all values
5. Add `b[f]`
6. Store in `Y[n, f, i, j]`

### 3) Padding Details

- Use zero-padding on height and width dimensions only.
- Do **not** pad batch or channel dimensions.
- If `P=0`, skip padding branch to keep logic clean.

### 4) Common Bugs to Avoid

- Mixing `NHWC` and `NCHW` formats
- Wrong output shape formula
- Off-by-one errors in loop ranges
- Forgetting bias per output channel
- Using integer arrays (must use float for math)

### 5) Minimal Verification Strategy

Before full training, verify forward pass with tiny controlled cases:

- Case A: `N=1, C_in=1, H=W=4, K=3, C_out=1, stride=1, pad=0`
  - Check exact numeric output manually.
- Case B: same but `pad=1`
  - Check output shape stays `4x4`.
- Case C: multi-channel input (`C_in=3`)
  - Ensure summation across channels is correct.

---

## Phase 3.3 — Convolution Backward Pass (Theory Notes)

### 📖 Objective

Given upstream gradient `dout` from next layer, compute:

1. `dW` — gradient for filters
2. `db` — gradient for biases
3. `dX` — gradient to pass to previous layer

All shapes use `NCHW`.

---

### 1) Forward reminder (single output element)

For sample `n`, filter `f`, output location `(i,j)`:

$$
out[n,f,i,j] = \sum\left(X_{patch} \odot W[f]\right) + b[f]
$$

where patch location is set by stride:

- `h_start = i * S_h`, `h_end = h_start + K_h`
- `w_start = j * S_w`, `w_end = w_start + K_w`

Backward just reverses this dependency.

---

### 2) Bias gradient `db`

`b[f]` is added to **every** output position of filter `f`, for every sample.

So:

$$
db[f] = \sum_{n,i,j} dout[n,f,i,j]
$$

This is usually the easiest gradient.

---

### 3) Weight gradient `dW`

Each weight value in `W[f]` contributed to many outputs.
For each output position, contribution is input patch scaled by `dout_value`.

So conceptually:

$$
dW[f] \; += \; X_{patch} \times dout[n,f,i,j]
$$

accumulated over all `n,i,j`.

Interpretation: if `dout` is large positive, increase weights in the direction of patch; if negative, opposite.

---

### 4) Input gradient `dX_padded`

Each input pixel influences multiple overlapping output cells.
So its gradient is a **sum of many routes** (chain rule accumulation).

For each output location:

$$
dX_{padded}[n,:,h\_start:h\_end,w\_start:w\_end] \; += \; W[f] \times dout[n,f,i,j]
$$

Why accumulation? Because one input region is reused by many sliding windows.

---

### 5) Padding handling in backward

If forward used padding, backward first builds gradient in padded space (`dX_padded`).
Final step is to crop back to original input size:

- If `P_h,P_w > 0`: return center region only
- If no padding: `dX = dX_padded`

This ensures returned `dX` has same shape as original `X`.

---

### 6) Stride effect in backward indexing

Stride does **not** change kernel size; it changes where gradients are written/read:

- forward mapping uses `h_start = i*S_h`, `w_start = j*S_w`
- backward must use the same mapping

With `S=2`, only every second spatial position is touched in mapping.

---

### 7) Practical implementation order (recommended)

1. Initialize `dW`, `db`, `dX_padded` as zeros
2. Loop `n, f, i, j`
3. `db[f] += dout_value`
4. `dW[f] += patch * dout_value`
5. `dX_padded[...] += W[f] * dout_value`
6. Crop `dX` from `dX_padded` if padded

---

### 8) Shape checklist (quick)

- `dout`: `(N, C_out, H_out, W_out)`
- `dW`: `(C_out, C_in, K_h, K_w)`
- `db`: `(C_out,)`
- `dX`: `(N, C_in, H, W)`

If any shape differs, stop and debug before running training.

---

## Phase 3.4 — Pooling Layers

### 📖 Theory

Pooling is a downsampling operation applied **independently per channel**.
It reduces spatial size (`H`, `W`) while keeping channel count `C` unchanged.

Why this helps:

1. Reduces computation and memory
2. Makes features more robust to small shifts/translations
3. Helps regularization by removing tiny noisy details

Input/Output convention (NCHW):

- Input: `(N, C, H, W)`
- Output: `(N, C, H_out, W_out)`

For pool window `(P_h, P_w)` and stride `(S_h, S_w)`:

$$
H_{out} = \left\lfloor \frac{H - P_h}{S_h} \right\rfloor + 1,
\qquad
W_{out} = \left\lfloor \frac{W - P_w}{S_w} \right\rfloor + 1
$$

Two common pooling types:

1. **MaxPool**
    - Forward: output the maximum value in each window
    - Backward: gradient flows only to the max element position (argmax)

2. **AvgPool**
    - Forward: output the average value in each window
    - Backward: gradient is evenly distributed to all elements in the window

Example (`2x2`, stride `2`, single channel):

Input
$$
\begin{bmatrix}
1 & 3 & 2 & 0 \\
4 & 6 & 5 & 1 \\
7 & 2 & 9 & 8 \\
3 & 1 & 4 & 2
\end{bmatrix}
$$

MaxPool output
$$
\begin{bmatrix}
6 & 5 \\
7 & 9
\end{bmatrix}
$$

AvgPool output
$$
\begin{bmatrix}
3.5 & 2.0 \\
3.25 & 5.75
\end{bmatrix}
$$

---

## Phase 3.5 (Flatten Layer)

### 📖 Theory

`Flatten` converts feature maps from 4D tensor to 2D tensor for Dense layers.

- Before flatten: `(N, C, H, W)`
- After flatten: `(N, C*H*W)`

Why needed:

1. Conv/Pool layers output spatial tensors
2. Dense layers expect vectors per sample
3. Flatten is the bridge between CNN blocks and classifier head

### Forward and Backward intuition

1. **Forward**: reshape each sample from `(C,H,W)` to 1D length `C*H*W`
2. **Backward**: reshape gradient back to original cached shape `(N,C,H,W)`

No parameters to learn (`Flatten` has no weights/bias).

---

### Minimal implementation checklist (no code)

1. In `forward(X)`, cache original input shape
2. Return reshaped output `(N, -1)`
3. In `backward(dout)`, reshape back to cached shape
4. Add shape validation for safety

---

### Tiny test expectations

If input shape is `(2, 3, 4, 4)`:

- Forward output shape should be `(2, 48)`
- Backward input gradient shape should be `(2, 3, 4, 4)`

---

## Phase 4.1 (Optimizers: SGD Theory)

We now have core CNN building blocks done:

1. Conv2D (forward/backward)
2. MaxPool2D (forward/backward)
3. AvgPool2D (forward/backward)
4. Flatten (forward/backward)

Next, we study parameter update rules (SGD first), then connect all layers into one training pipeline.

### 📖 Theory — SGD (Stochastic Gradient Descent)

After backprop, each trainable layer has parameter gradients:

- weights gradient (`dW`)
- bias gradient (`db`)

SGD updates parameters by moving **opposite** the gradient direction:

$$
W \leftarrow W - \eta \cdot dW,
\qquad
b \leftarrow b - \eta \cdot db
$$

where $\eta$ is learning rate.

---

### Why this works

Gradient points to the direction of increasing loss.
Subtracting it moves parameters toward lower loss.

If learning rate is:

- too large → unstable / oscillating loss
- too small → very slow learning

---

### “Stochastic” meaning in practice

We usually compute gradients on a mini-batch, not full dataset:

1. sample batch
2. forward
3. backward
4. update parameters immediately

This introduces noise, but often helps optimization escape poor local patterns.

---

### Minimal implementation plan (no code)

Create an SGD helper with one method `step(layers)`:

1. Iterate over model layers
2. If layer has trainable params (`w`, `b`) and gradients (`dW`, `db`)
3. Apply update rule using `learning_rate`
4. (Optional) add gradient clipping later if unstable

---

## Phase 4.2 Status 

Implemented in `src/train.py`:

1. Mini-batch shuffle per epoch
2. Forward: Conv → ReLU → MaxPool → Flatten → Dense → Softmax
3. Loss: Cross-Entropy
4. Backward chain through all layers
5. SGD update per batch
6. Epoch metrics (loss, train/test accuracy)

---

## Phase 4.3 — Learning Rate Scheduling (Theory)

### Why schedule LR?

High LR is useful early for fast progress; lower LR later helps stable convergence.

Simple schedules for this project:

1. **Step decay:** reduce LR every fixed number of epochs
2. **Exponential decay:** `lr = lr0 * gamma^epoch`
3. **Plateau decay (manual):** reduce LR if validation metric stalls

Recommended starter:

- Start `lr=0.01`
- Every 5 epochs: `lr *= 0.5`

---

## Phase 4.4 — Regularization (Theory)

Goal: reduce overfitting and improve generalization.

Useful first methods:

1. **L2 weight decay**
    - Add penalty on large weights
    - Update rule includes `lambda * W`
2. **Data augmentation** (very important for CIFAR-10)
    - random crop, horizontal flip, small color jitter
3. **Dropout** (optional later)
    - easier in Dense stage, harder in from-scratch CNN loops

Recommended priority for this repo:

1. Data augmentation
2. L2 weight decay
3. (Optional) dropout later

---

## Phase 5 — Putting It All Together

### 5.1 Full CNN Architecture (Practical target)

Current model is small. For stronger accuracy target, move to:

`Conv(16)-ReLU-Pool -> Conv(32)-ReLU-Pool -> Flatten -> Dense(128)-ReLU -> Dense(10)-Softmax`

### 5.2 Training Loop (Done baseline, needs scaling)

Already implemented baseline loop; next improve runtime/stability and training duration.

### 5.3 Evaluation & Metrics

Track:

1. Train loss
2. Train accuracy
3. Test accuracy
4. (Optional) confusion matrix per class

### 5.4 Visualization

Plot per epoch:

1. Loss curve
2. Train/Test accuracy curves
3. (Optional) first-layer filters

### 5.5 Experimentation Plan

Tune in this order:

1. Learning rate + schedule
2. Batch size
3. Model width (`out_channels`, hidden size)
4. Data augmentation
5. Weight decay

---

## Final Pre-Training Plan (Before long run)

1. Keep quick run for sanity checks
2. Switch to medium run (10k train, 2k test, 5 epochs)
3. Add LR schedule
4. Add data augmentation
5. Run full training and compare curves

After this, we can target stronger accuracy and prepare stable notebook inference/demo.