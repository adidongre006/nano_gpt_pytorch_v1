# Character-Level GPT Language Model (Trained from Scratch)

A decoder-only, GPT-style Transformer implemented in PyTorch and trained **from scratch** on the Tiny Shakespeare corpus to perform next-character prediction. This is an educational/from-scratch implementation — it is **not** a pretrained GPT model and is not fine-tuned from any existing checkpoint; every weight is learned starting from random initialization on this single dataset.

The project implements the core building blocks of a GPT-style Transformer by hand — causal self-attention, multi-head attention, position-wise feed-forward layers, pre-norm residual blocks, and an autoregressive sampling loop — making it a compact reference for understanding how GPT models work internally.

**Technologies used:** Python, PyTorch (`torch.nn`, `torch.nn.functional`), Jupyter Notebook.

> **From-scratch vs. pretrained:** This model is trained entirely on Tiny Shakespeare with randomly initialized weights. It is a small (~10.79M parameter), character-level model intended for learning/demonstration purposes, not a general-purpose language model comparable to pretrained GPT-2/GPT-3-class systems.

## Table of Contents

- [Character-Level GPT Language Model (Trained from Scratch)](#character-level-gpt-language-model-trained-from-scratch)
  - [Table of Contents](#table-of-contents)
  - [Model Overview](#model-overview)
  - [Architecture](#architecture)
  - [Model Configuration](#model-configuration)
  - [Dataset and Tokenization](#dataset-and-tokenization)
  - [Training and Loss Calculation](#training-and-loss-calculation)
  - [Training and Validation Loss](#training-and-validation-loss)
  - [Model Parameter Analysis](#model-parameter-analysis)
  - [Saving the Model](#saving-the-model)
  - [Loading the Trained Model](#loading-the-trained-model)
  - [Text Generation](#text-generation)
  - [Installation and Execution](#installation-and-execution)
  - [Results and Limitations](#results-and-limitations)
  - [Future Improvements](#future-improvements)

## Model Overview

| Property | Value |
|---|---|
| Model architecture | Decoder-only Transformer (GPT-style), pre-norm residual blocks, causal self-attention |
| Framework | PyTorch |
| Tokenization | Character-level (custom vocabulary built from the training text) |
| Vocabulary size | Not reported — not printed in the notebook output (determined at runtime as the number of unique characters in the training text) |
| Context length (`block_size`) | 256 |
| Embedding dimension (`n_embd`) | 384 |
| Attention heads (`n_head`) | 6 |
| Transformer layers (`n_layer`) | 6 |
| Total parameters | 10,788,929 (~10.79M), verified from notebook output |
| Training objective | Next-character prediction (autoregressive language modeling), cross-entropy loss |
| Optimizer | AdamW, learning rate `3e-4` |
| Dataset | Tiny Shakespeare (`input_tiny_shakespear.txt`) |

## Architecture

The model is a stack of `n_layer` identical Transformer blocks operating on learned token and positional embeddings. Each block applies causal multi-head self-attention followed by a position-wise feed-forward network, both wrapped in pre-norm residual connections (`LayerNorm` → sublayer → add). A final `LayerNorm` and a linear head project the last hidden state to vocabulary-sized logits.

![alt text](image-1.png)

**Component summary:**

- **Token & positional embeddings** — `nn.Embedding(vocab_size, n_embd)` and `nn.Embedding(block_size, n_embd)` are summed to give each token a content representation plus a position signal.
- **Causal self-attention (`Head`)** — each head projects the input to key/query/value vectors (linear, no bias), computes scaled dot-product attention, and masks future positions with a lower-triangular buffer (`tril`) so a token can only attend to itself and earlier tokens.
- **Multi-head attention** — `n_head` independent `Head` instances run in parallel; their outputs are concatenated and projected back to `n_embd` via a linear layer, followed by dropout.
- **Feed-forward network** — a 2-layer MLP (`n_embd → 4·n_embd → n_embd`) with a ReLU non-linearity and dropout, applied identically at every position.
- **Transformer block** — combines attention and feed-forward with pre-norm residual connections: `x = x + sa(ln1(x))`, `x = x + ffwd(ln2(x))`.
- **LM head** — a final `LayerNorm` followed by a linear layer maps hidden states to a distribution over the vocabulary.

Causal self-attention is implemented as described (masking is real and verified in the source); no components beyond those listed above (e.g., no rotary embeddings, no KV-cache) are present in this implementation.

## Model Configuration

**Architecture parameters**

| Parameter | Value | Description |
|---|---|---|
| `vocab_size` | Not reported | Number of unique characters in the training text; computed at runtime, not printed |
| `block_size` | 256 | Maximum context length (tokens attended to per prediction) |
| `n_embd` | 384 | Embedding / residual-stream dimension |
| `n_head` | 6 | Number of attention heads per block |
| `n_layer` | 6 | Number of stacked Transformer blocks |
| `head_size` (derived) | 64 | `n_embd // n_head` = 384 / 6 |
| `dropout` | 0.2 | Dropout probability applied in attention and feed-forward sublayers |

**Training hyperparameters**

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | 32 | Sequences processed per training step (code comment notes 32/16 to avoid CUDA OOM) |
| `max_iters` | 5000 | Total optimizer steps |
| `eval_interval` | 500 | Steps between loss evaluations |
| `eval_iters` | 200 | Batches averaged per loss estimate |
| `learning_rate` | 3e-4 | AdamW learning rate |
| `optimizer` | AdamW | `torch.optim.AdamW(model.parameters(), lr=learning_rate)` |
| `device` | `cuda` if available, else `cpu` | Verified run used CUDA on an NVIDIA GeForce RTX 2050 |
| Random seed | 1337 | `torch.manual_seed(1337)` |

## Dataset and Tokenization

The model is trained on **Tiny Shakespeare**, loaded from a local file `input_tiny_shakespear.txt`. Exact file size (character count) is not reported/printed in the notebook.

- **Vocabulary creation:** the set of unique characters in the raw text is sorted to form a fixed character vocabulary; `vocab_size` is the length of this set.
- **Encoding/decoding:** two dictionaries map characters to integer indices and back (`stoi`, `itos`).
- **Train/validation split:** the encoded corpus is split 90% / 10% by position (first 90% train, remaining 10% validation) — no shuffling across the split.
- **Input-target preparation:** `get_batch` draws random starting indices, then builds input blocks `x` of length `block_size` and target blocks `y` as the same window shifted one character to the right (standard next-token prediction setup).

```python
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]            # encoder: string -> list[int]
decode = lambda l: ''.join([itos[i] for i in l])   # decoder: list[int] -> string
```

## Training and Loss Calculation

The model is trained with the standard **next-token prediction** objective: at every position, predict the next character given all preceding characters in the context window.

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log P(x_i \mid x_{<i})
$$

where $N = B \times T$ is the number of tokens in a batch (`batch_size × block_size`), and $P(x_i \mid x_{<i})$ is the model's predicted probability for the true next character.

```python
# inside GPTLanguageModel.forward
B, T, C = logits.shape
logits = logits.view(B * T, C)
targets = targets.view(B * T)
loss = F.cross_entropy(logits, targets)
```

**Training loop** (verified from the notebook):

| Step | Action |
|---|---|
| 1 | Every `eval_interval` steps (or on the last step), compute averaged train/val loss via `estimate_loss()` and print it |
| 2 | Sample a random training batch with `get_batch('train')` |
| 3 | Forward pass: `logits, loss = model(xb, yb)` |
| 4 | `optimizer.zero_grad(set_to_none=True)` |
| 5 | `loss.backward()` |
| 6 | `optimizer.step()` |

This repeats for `max_iters = 5000` steps using AdamW at a fixed learning rate of `3e-4` (no learning-rate schedule is implemented).

----
## Training and Validation Loss

![alt text](image.png)


Training loss measures next-character prediction error on randomly sampled training batches; validation loss measures the same on held-out data never used for gradient updates, indicating generalization.

**Recorded losses (verified from notebook output):**

| Step | Train Loss | Val Loss |
|---|---|---|
| 0 | 4.2291 | 4.2354 |
| 500 | 1.9557 | 2.0530 |
| 1000 | 1.5232 | 1.7261 |
| 1500 | 1.3788 | 1.5937 |
| 2000 | 1.2992 | 1.5383 |
| 2500 | 1.2388 | 1.5142 |
| 3000 | 1.1981 | 1.4931 |
| 3500 | 1.1592 | 1.4957 |
| 4000 | 1.1223 | 1.5029 |
| 4500 | 1.0901 | 1.4865 |
| 4999 | 1.0579 | 1.4903 |

Train loss decreases steadily throughout training, while validation loss flattens out around **~1.49** after step ~3000 and no longer tracks the training loss downward. This growing train/val gap is a sign of mild overfitting in the later training steps, even though validation loss never diverges sharply upward.

The notebook includes a "Plotting the train v/s loss graph" section, but the corresponding code cell is empty — no loss curve image was generated. No `plots/loss_curve.png` (or equivalent) exists in the project. The snippet below shows how the curve could be produced, assuming loss history is captured during training (the current training loop only prints losses; it does not store them in a list):

```python
import matplotlib.pyplot as plt

# Requires collecting these during training, e.g.:
# train_losses.append(losses['train']); val_losses.append(losses['val'])
steps = list(range(0, max_iters, eval_interval)) + [max_iters - 1]

plt.plot(steps, train_losses, label='Train Loss')
plt.plot(steps, val_losses, label='Validation Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()
```

## Model Parameter Analysis

Trainable parameters are the weights and biases updated by the optimizer during backpropagation. In this project, no layers are frozen, so all parameters are trainable.

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

print(f"Total: {total_params:,} | Trainable: {trainable_params:,} | Non-trainable: {non_trainable_params:,}")
```

**Verified from notebook output:**

```
10.788929 M parameters
```

This corresponds to **10,788,929** total parameters, all trainable (no non-trainable/frozen parameters in this implementation).

**Parameter memory (weights only, FP32):** 10,788,929 × 4 bytes ≈ **41.2 MiB (~43.2 MB)**. This covers only the stored weight values — it does **not** include gradients, AdamW's two per-parameter moment estimates (which roughly triple the optimizer-related memory beyond the weights alone), or activation memory, which scales with batch size and sequence length. Total training memory is therefore substantially higher than the raw parameter size.

## Saving the Model

**Not implemented in the current notebook** — no `torch.save` call is present, so trained weights are not currently persisted to disk. The snippet below is a suggested addition, using the actual class/variable names from this project (`model.state_dict()`, the real hyperparameters, and the `stoi`/`itos` tokenizer mappings, which must be saved alongside the weights since the character vocabulary is dataset-specific):

```python

import os

os.makedirs("checkpoints",exist_ok=True)

checkpoint = {
    'model_stete_dict':model.state_dict(),
    'model_config':{
        "block_size":block_size,
        "n_embd":n_embd,
        "n_head":n_head,
        "n_layer":n_layer,
        "dropout":dropout,
        "vocab_size":vocab_size,
    },
    "chars":chars,
    "stoi":stoi,
    "itos":itos,
}

torch.save(checkpoint,"checkpoints/nano_gpt_model.pth")
print("Model saved")
```

Saving only `model.state_dict()` persists weights for inference. Saving the full checkpoint above (config, tokenizer mappings, and — if resuming training — `optimizer.state_dict()`) is required to reconstruct the exact model and continue training or run inference outside the current session.

## Loading the Trained Model

The `GPTLanguageModel` class currently lives inline in the notebook and its constructor takes **no configuration arguments** — it reads `vocab_size`, `n_embd`, `n_head`, `n_layer`, and `block_size` directly from module-level global variables. For a clean, reusable inference workflow, this class should be moved into a `model.py` module (and ideally refactored to accept these as constructor arguments instead of relying on globals).

Given the current (global-variable-based) constructor, loading looks like this:

```python
## imports 
import torch
from model_file import GPTLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## loading the checkpoints 
checkpoint = torch.load(
    "checkpoints/nano_gpt_model.pth",
    map_location=device,
    weights_only=True,
)

config = checkpoint["model_config"]

block_size = config["block_size"]
n_embd = config["n_embd"]
n_head = config["n_head"]
n_layer = config["n_layer"]
dropout = config["dropout"]
vocab_size = config["vocab_size"]

## Restore Vocabulary

chars = checkpoint["chars"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

## encoder and decoders

def encode(text):
    return [stoi[c] for c in text]

def decode(text):
    return "".join(itos[i] for i in text)


## Creating the model and load trained weigths

model = GPTLanguageModel().to(device)

model.load_state_dict(checkpoint["model_stete_dict"])

model.eval()

```

## Text Generation

Generation is autoregressive: at## Restore the settings before creating the model 


 each step the model is run on the current context (cropped to the last `block_size` tokens), the logits for the final position are converted to a probability distribution via softmax, the next character is sampled from that distribution with `torch.multinomial`, and it is appended to the sequence before repeating.

```python
## Generating the text with loaded model

prompt="hello world"

input_tokens = torch.tensor(
    [encode(prompt)],
    dtype= torch.long,
    device=device,
)

with torch.no_grad():
    output_tokens = model.generate(
        input_tokens,
        max_new_tokens=10
    )

print(decode(output_tokens[0].tolist())) 

```

**Output** (prompt `"hello world"`, `max_new_tokens=10`):

```
hello world a thing, 
```

Because `prompt_len` is computed as the **word count** of the prompt (3 words) rather than a deliberately chosen generation length, only 3 new characters were sampled and appended after the prompt in this test — the output is not a substantial generated passage.

<!-- ## Project Structure

Only the notebook (reviewed here via its exported HTML, `GPT_model_v1.html`) and the dataset filename referenced in code (`input_tiny_shakespear.txt`) are confirmed to exist. No `model.py`, `requirements.txt`, checkpoint files, or `plots/` directory are present in the project as reviewed. The tree below is a **suggested structure** for organizing the repository:

```
.
├── GPT_model_v1.ipynb          # confirmed: main notebook (source of this README)
├── input_tiny_shakespear.txt   # confirmed: training corpus, referenced in code
├── model.py                    # suggested: extracted Head/MultiHeadAttention/Block/GPTLanguageModel classes
├── gpt_checkpoint.pt           # suggested: saved weights + config + tokenizer (not currently produced)
├── plots/
│   └── loss_curve.png          # suggested: rendered train/val loss plot (not currently produced)
├── requirements.txt            # suggested: pinned dependencies (not currently present)
└── README.md
``` -->

## Installation and Execution

**Dependencies (verified from imports):** `torch` (`torch.nn`, `torch.nn.functional`). No `requirements.txt` is provided in the project; a version pin was not printed, so install the latest compatible PyTorch build for your environment.

```bash
pip install torch
```

Run all cells in order — hyperparameters, dataset loading, model/class definitions, model instantiation, optimizer creation, and the training loop must execute sequentially since later cells depend on globals defined earlier (e.g., `vocab_size`, `model`, `m`).

**Inference:** the project does not currently expose a standalone inference script or saved checkpoint. Text generation is only demonstrated inline, after training, via the `m.generate(...)` call shown in [Text Generation](#text-generation). See [Saving the Model](#saving-the-model) and [Loading the Trained Model](#loading-the-trained-model) for a suggested path to a reusable inference workflow.

## Results and Limitations

**Results (verified from notebook output):**

| Metric | Value |
|---|---|
| Total parameters | 10,788,929 (~10.79M) |
| Training iterations | 5,000 |
| Final training loss (step 4999) | 1.0579 |
| Final validation loss (step 4999) | 1.4903 |
| Validation perplexity (computed as $e^{1.4903}$) | ≈ 4.44 |
| Training device | CUDA — NVIDIA GeForce RTX 2050 |
| Dataset | Tiny Shakespeare (character-level) |

**Limitations:**

- No checkpoint saving/loading is implemented — the model must be retrained from scratch to reproduce results or generate text in a new session.
- `GPTLanguageModel`'s constructor takes no arguments and depends on module-level globals, limiting portability outside the notebook.
- The loss-curve plotting section is present as a header only; no plot is generated or saved.
- The one generation example produced only 3 new characters due to `prompt_len` being computed as a word count rather than an intentional generation length; generation quality is not otherwise evaluated.
- Character-level tokenization means the vocabulary and effective context are tied to raw characters rather than sub-word units, and the trained vocabulary is specific to this dataset.
- Validation loss plateaus (~1.49) while training loss keeps decreasing after step ~3000, indicating the onset of mild overfitting; no early stopping or regularization changes were applied to address this.
- No `requirements.txt` or environment/version pinning is included.

## Future Improvements

| Area | Improvement |
|---|---|
| Persistence | Implement `torch.save`/`torch.load` checkpointing (weights, config, `stoi`/`itos`) as shown in [Saving the Model](#saving-the-model) |
| Reusability | Refactor `GPTLanguageModel` into `model.py` with a parameterized constructor instead of relying on global variables |
| Evaluation | Track and plot the train/validation loss curve; add perplexity tracking during training, not just post-hoc |
| Generation | Fix `prompt_len` to reflect intended generation length (characters) rather than prompt word count; evaluate longer generated samples |
| Tokenization | Explore sub-word/BPE tokenization (e.g., `tiktoken`) for improved sample efficiency over character-level encoding |
| Training | Add a learning-rate schedule (warmup/decay) on top of the current fixed `3e-4` AdamW rate; investigate regularization to close the train/val loss gap observed after step ~3000 |
| Tooling | Add `requirements.txt` / environment pinning for reproducibility |

<!-- ## License

Not reported — no license file or license section was found in the reviewed project. Add a license (e.g., MIT, Apache-2.0) before publishing publicly. -->