---
name: pytorch-tensorflow-conversion
description: Use when converting Python code between PyTorch and TensorFlow/Keras frameworks. Triggers include requests to "convert to TensorFlow", "rewrite in PyTorch", "port this model", migrate training loops, or translate layer definitions between frameworks.
---

# PyTorch ↔ TensorFlow Conversion

## Overview

Systematic approach for converting deep learning code between PyTorch and TensorFlow/Keras. Covers layers, training loops, data formats, and common pitfalls.

## Critical First Check: Data Format

**PyTorch default:** NCHW (batch, channels, height, width)
**TensorFlow default:** NHWC (batch, height, width, channels)

**Options when converting:**

| Approach | When to Use |
|----------|-------------|
| Transpose data | Quick fix, inference only |
| Set `data_format='channels_first'` | Preserving PyTorch weights |
| Rewrite for NHWC | Production TensorFlow code |

```python
# PyTorch -> TensorFlow: Transpose input
x_tf = np.transpose(x_pt, (0, 2, 3, 1))  # NCHW -> NHWC

# TensorFlow -> PyTorch: Transpose input
x_pt = np.permute(x_tf, (0, 3, 1, 2))    # NHWC -> NCHW
```

## Quick Reference: Layer Mappings

### Convolution & Pooling

| PyTorch | TensorFlow/Keras |
|---------|------------------|
| `nn.Conv2d(in, out, k, stride, padding)` | `Conv2D(out, k, strides, padding)` |
| `nn.Conv1d` | `Conv1D` |
| `nn.Conv3d` | `Conv3D` |
| `nn.MaxPool2d(k, stride)` | `MaxPool2D(pool_size=k, strides=stride)` |
| `nn.AvgPool2d` | `AveragePooling2D` |
| `nn.AdaptiveAvgPool2d((H, W))` | `GlobalAveragePooling2D` (if (1,1)) or `tf.keras.layers.Resizing` |
| `nn.functional.pad` | `tf.pad` or `ZeroPadding2D` |
───
**Padding difference:**
- PyTorch: `padding=1` means 1 pixel on all sides
- TensorFlow: `padding='same'` (auto) or `padding='valid'` (none)

### Normalization

| PyTorch | TensorFlow/Keras |
|---------|------------------|
| `nn.BatchNorm2d(channels)` | `BatchNormalization()` |
| `nn.LayerNorm(dim)` | `LayerNormalization()` |
| `nn.GroupNorm(num_groups, channels)` | `tf.keras.layers.GroupNormalization(groups=num_groups)` |
| `nn.InstanceNorm2d` | `tf.keras.layers.InstanceNormalization` (add-on) |

**Critical:** TensorFlow BatchNorm needs `training` parameter in `call()`:
```python
# PyTorch
x = self.bn(x)

# TensorFlow
x = self.bn(x, training=training)  # Must pass training flag
```

### Activation & Linear

| PyTorch | TensorFlow/Keras |
|---------|------------------|
| `nn.ReLU()` | `ReLU()` or `activation='relu'` |
| `nn.LeakyReLU(0.1)` | `LeakyReLU(alpha=0.1)` |
| `nn.GELU()` | `GELU()` |
| `nn.Sigmoid()` | `Sigmoid()` |
| `nn.Softmax(dim=1)` | `Softmax()` |
| `nn.Linear(in, out)` | `Dense(out)` (no input dim needed) |
| `nn.Dropout(p=0.5)` | `Dropout(0.5)` |

### Recurrent Layers

| PyTorch | TensorFlow/Keras |
|---------|------------------|
| `nn.LSTM(hidden, batch_first=True)` | `LSTM(hidden, return_sequences=True)` |
| `nn.GRU(hidden)` | `GRU(hidden, return_sequences=True)` |
| `nn.RNN(hidden)` | `SimpleRNN(hidden)` |

### Embedding & Transformers

| PyTorch | TensorFlow/Keras |
|---------|------------------|
| `nn.Embedding(vocab, dim)` | `Embedding(vocab, dim)` |
| `nn.Transformer` | `tf.keras.layers.MultiHeadAttention` |
| `nn.MultiheadAttention` | `MultiHeadAttention` |

## Quick Reference: Training Components

### Loss Functions

| PyTorch | TensorFlow/Keras | Note |
|---------|------------------|------|
| `nn.CrossEntropyLoss()` | `SparseCategoricalCrossentropy(from_logits=True)` | TF expects labels first |
| `nn.BCELoss()` | `BinaryCrossentropy()` | - |
| `nn.BCEWithLogitsLoss()` | `BinaryCrossentropy(from_logits=True)` | - |
| `nn.MSELoss()` | `MeanSquaredError()` | - |
| `nn.L1Loss()` | `MeanAbsoluteError()` | - |
| `nn.NLLLoss()` | `SparseCategoricalCrossentropy(from_logits=False)` | Expects log-probs |
| `nn.KLDivLoss()` | `KLDivergence()` | - |

**Argument order difference:**
```python
# PyTorch: prediction first
loss = criterion(outputs, labels)

# TensorFlow: ground truth first
loss = loss_fn(labels, outputs)  # y_true, y_pred
```

### Optimizers

| PyTorch | TensorFlow/Keras |
|---------|------------------|
| `optim.SGD(params, lr, momentum)` | `SGD(learning_rate=lr, momentum=momentum)` |
| `optim.Adam(params, lr)` | `Adam(learning_rate=lr)` |
| `optim.AdamW(params, lr)` | `AdamW(learning_rate=lr)` |
| `optim.RMSprop(params, lr)` | `RMSprop(learning_rate=lr)` |
| `optim.Adagrad(params, lr)` | `Adagrad(learning_rate=lr)` |

### Learning Rate Schedulers

| PyTorch | TensorFlow/Keras |
|---------|------------------|
| `optim.lr_scheduler.StepLR(opt, step, gamma)` | `tf.keras.optimizers.schedules.ExponentialDecay` |
| `optim.lr_scheduler.CosineAnnealingLR` | `tf.keras.optimizers.schedules.CosineDecay` |
| `optim.lr_scheduler.ReduceLROnPlateau` | `tf.keras.callbacks.ReduceLROnPlateau` |
| `optim.lr_scheduler.OneCycleLR` | Custom or `tfa.optimizers.CyclicalLearningRate` |

## Quick Reference: Training Loop Patterns

### PyTorch → TensorFlow

```python
# === PyTorch ===
model.train()
for epoch in range(epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

# === TensorFlow ===
for epoch in range(epochs):
    for x, y in train_dataset:
        with tf.GradientTape() as tape:
            output = model(x, training=True)
            loss = loss_fn(y, output)  # y_true first!
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

### TensorFlow → PyTorch

```python
# === TensorFlow ===
with tf.GradientTape() as tape:
    output = model(x, training=True)
    loss = loss_fn(y, output)
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

# === PyTorch ===
optimizer.zero_grad()
output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()
```

## Quick Reference: Mode Switching

| PyTorch | TensorFlow/Keras |
|---------|------------------|
| `model.train()` | N/A (pass `training=True` in call) |
| `model.eval()` | N/A (pass `training=False` in call) |
| `with torch.no_grad():` | `@tf.function` or `tf.stop_gradient` |
| `x.requires_grad_(False)` | `tf.stop_gradient(x)` or `trainable=False` |
| `torch.cuda.is_available()` | `tf.config.list_physical_devices('GPU')` |
| `x.to(device)` | Automatic (TF handles device placement) |

## Quick Reference: Data Loading

| PyTorch | TensorFlow/Keras |
|---------|------------------|
| `DataLoader(dataset, batch_size, shuffle)` | `tf.data.Dataset.batch().shuffle()` |
| `TensorDataset(x, y)` | `tf.data.Dataset.from_tensor_slices((x, y))` |
| `dataset[i]` | `dataset.skip(i).take(1)` or `dataset.batch(1)` |

```python
# PyTorch
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# TensorFlow equivalent
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
```

## Quick Reference: Model Definition Styles

### Three approaches in TensorFlow

```python
# 1. Subclassing (closest to PyTorch nn.Module)
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(64)
        self.dense2 = Dense(10)

    def call(self, x, training=False):
        x = self.dense1(x)
        return self.dense2(x)

# 2. Functional API (recommended for most cases)
inputs = Input(shape=(784,))
x = Dense(64)(inputs)
outputs = Dense(10)(x)
model = Model(inputs, outputs)

# 3. Sequential (simplest)
model = Sequential([
    Dense(64, input_shape=(784,)),
    Dense(10)
])
```

## Quick Reference: Weight Initialization

| PyTorch | TensorFlow/Keras |
|---------|------------------|
| `nn.init.kaiming_normal_` | `tf.keras.initializers.HeNormal()` |
| `nn.init.xavier_normal_` | `tf.keras.initializers.GlorotNormal()` |
| `nn.init.xavier_uniform_` | `tf.keras.initializers.GlorotUniform()` |
| `nn.init.constant_` | `tf.keras.initializers.Constant()` |

```python
# PyTorch
nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

# TensorFlow
layer = Dense(64, kernel_initializer='he_normal')
```

## Quick Reference: Saving & Loading

| PyTorch | TensorFlow/Keras |
|---------|------------------|
| `torch.save(model.state_dict(), 'model.pt')` | `model.save_weights('model.h5')` |
| `model.load_state_dict(torch.load('model.pt'))` | `model.load_weights('model.h5')` |
| `torch.save(model, 'full.pt')` | `model.save('full_model')` |
| `torch.jit.script(model)` | `@tf.function` + `saved_model` |

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Wrong data format | Weird output shapes, NaN losses | Transpose to NCHW/NHWC or set `data_format` |
| Loss argument order | Shape mismatch errors | TF: `(y_true, y_pred)`, PyTorch: `(y_pred, y_true)` |
| Missing training flag | BatchNorm uses wrong stats | Pass `training=True/False` in TF |
| Conv padding mismatch | Output size differs | PyTorch `padding=1` ≠ TF `padding='same'` |
| Wrong Dense input dim | TF Dense ignores input, uses output only | `Dense(output_dim)` not `Dense(in, out)` |
| Forgetting optimizer.zero_grad() | Loss doesn't decrease | Reset gradients each iteration |
| Gradient tape in wrong scope | No gradients computed | Keep tape around forward pass |

## Conversion Checklist

Before converting, identify:

- [ ] Data format (NCHW vs NHWC) - decide strategy
- [ ] Model architecture style (subclassing vs functional vs sequential)
- [ ] Training loop style (custom vs `fit()`)
- [ ] Device handling needs (TF auto-handles, PyTorch explicit)

After converting, verify:

- [ ] Input/output shapes match
- [ ] Parameter counts match
- [ ] Loss values in same range
- [ ] Forward pass produces same (or transposed) results
- [ ] Training loop converges similarly

## Complete Example

### PyTorch Original

```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Usage
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for inputs, targets in train_loader:  # NCHW format
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

### TensorFlow Equivalent

```python
import tensorflow as tf
from tensorflow.keras import layers, models

class CNN(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = models.Sequential([
            layers.Conv2D(64, 3, padding='same'),  # filters first, no input channels
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(2),
            layers.Conv2D(128, 3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(2),
        ])
        self.classifier = models.Sequential([
            layers.Dropout(0.5),
            layers.Dense(num_classes),
        ])

    def call(self, x, training=False):
        x = self.features(x, training=training)  # Pass training to nested model
        x = layers.Flatten()(x)
        x = self.classifier(x, training=training)
        return x

# Usage
model = CNN()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

for inputs, targets in train_dataset:  # NHWC format (transpose if needed!)
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = loss_fn(targets, outputs)  # targets first!
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

**Key differences to note:**
1. Conv2D: `(filters, kernel_size)` not `(in_channels, out_channels, kernel_size)`
2. Loss: `(y_true, y_pred)` not `(y_pred, y_true)`
3. Training: Passed explicitly, not set via `model.train()`
4. Data: Ensure NHWC format for inputs
