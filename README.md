# PyTorch ↔ TensorFlow Converter

A Claude Code skill for converting deep learning code between PyTorch and TensorFlow/Keras frameworks.

## Installation

### Option 1: Manual Installation

```bash
# Create the skill directory
mkdir -p ~/.claude/skills/pytorch-tensorflow-conversion

# Copy the skill file
cp SKILL.md ~/.claude/skills/pytorch-tensorflow-conversion/
```

### Option 2: Clone and Link

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/pytorch-tensorflow-converter.git

# Link or copy to your skills directory
cp pytorch-tensorflow-converter/SKILL.md ~/.claude/skills/pytorch-tensorflow-conversion/
```

## Usage

Once installed, the skill activates automatically when you ask Claude to convert code between frameworks.

### Trigger Phrases

- "Convert this PyTorch code to TensorFlow"
- "Rewrite this in PyTorch"
- "Port this model to Keras"
- "Migrate this training loop to TensorFlow"
- "Translate these layer definitions"

### Example

```
You: Convert this PyTorch model to TensorFlow:

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

Claude: [Loads skill and provides accurate TensorFlow equivalent with data format notes]
```

## What's Covered

| Category | Details |
|----------|---------|
| **Layers** | Conv, Pooling, Normalization, Activation, Linear, Recurrent, Transformer |
| **Loss Functions** | CrossEntropy, BCE, MSE, KL Divergence, etc. |
| **Optimizers** | SGD, Adam, AdamW, RMSprop, Adagrad |
| **Training Loops** | GradientTape vs backward(), zero_grad pattern |
| **Data Handling** | DataLoader vs tf.data.Dataset, NCHW vs NHWC |
| **Model I/O** | Saving/loading weights and full models |
| **Common Pitfalls** | Data format, argument order, training flags |

## Key Features

- **Quick reference tables** for common layer and function mappings
- **Data format handling** (NCHW ↔ NHWC) with clear strategies
- **Complete examples** showing full model conversions
- **Common mistakes** section with symptoms and fixes
- **Conversion checklist** for before/after verification

## Requirements

- Claude Code CLI
- No additional dependencies (it's a documentation skill)

## Contributing

Found a missing mapping or have an improvement?

1. Fork the repository
2. Edit `SKILL.md`
3. Submit a pull request

## License

MIT License - feel free to use and modify.
