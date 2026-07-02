# 2.3MParams-LLM-From-Scratch-Python

<a href="https://colab.research.google.com/drive/1AlnGsNU3BauFhn3ZY6tk71G9ANUAQRtx?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

<img src="https://i.ibb.co/r56NHtM/1-ox3h-To-PFUWx-Aw-URx-YEXi-Gg-removebg-preview.png" alt="LLM Architecture Diagram">

Building a custom Large Language Model (LLM) is a significant undertaking that many major organizations like Google, Meta, and X are pursuing. They release various iterations of these models, ranging from 7 billion to 70 billion parameters. While there is a wealth of theoretical information available, practical step-by-step guides and code implementations are less common.

This project provides a detailed implementation of the LLaMA architecture from scratch, building upon the foundational work originally shared by Brian Kitano.

In this repository, I have implemented an LLM with 2.3 million parameters. A key feature of this implementation is that it does not require a high-end GPU for training. By following the LLaMA 1 Paper approach, this project demonstrates how to create a million-parameter model using a basic dataset and accessible hardware.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Understanding the Transformer Architecture of LLaMA](#understanding-the-transformer-architecture-of-llama)
  - [Pre-normalization Using RMSNorm](#pre-normalization-using-rmsnorm)
  - [SwiGLU Activation Function](#swiglu-activation-function)
  - [Rotary Embeddings (RoPE)](#rotary-embeddings-rope)
- [Setting the Stage](#setting-the-stage)
- [Data Preprocessing](#data-preprocessing)
- [Evaluation Strategy](#evaluation-strategy)
- [Setting Up a Base Neural Network Model](#setting-up-a-base-neural-network-model)
- [Replicating LLaMA Architecture](#replicating-llama-architecture)
  - [RMSNorm for pre-normalization](#rmsnorm-for-pre-normalization)
  - [Rotary Embeddings](#rotary-embeddings)
  - [SwiGLU activation function](#swiglu-activation-function)
- [Experimenting with hyperparameters](#experimenting-with-hyperparameters)
- [Saving Your Language Model (LLM)](#saving-your-language-model-llm)
- [Conclusion](#conclusion)

## Prerequisites

A basic understanding of object-oriented programming (OOP) and neural networks (NN) is required. Familiarity with PyTorch is also helpful for understanding the code implementation.

| Topic               | Video Link                                                |
|---------------------|-----------------------------------------------------------|
| OOP                 | [OOP Video](https://www.youtube.com/watch?v=Ej_02ICOIgs) |
| Neural Network      | [Neural Network Video](https://www.youtube.com/watch?v=Jy4wM2X21u0) |
| Pytorch             | [Pytorch Video](https://www.youtube.com/watch?v=V_xro1bcAuA) |

## Understanding the Transformer Architecture of LLaMA

Before implementing the model, it is essential to understand the LLaMA architecture. Below is a comparison between the vanilla transformer and the LLaMA approach.

<img src="https://cdn-images-1.medium.com/max/25620/1*nt-ydHhSVsaLXq_HZRaLQA.png" alt="Difference between Transformers and Llama architecture" style="width: 50%;">
(Llama architecture by Umar Jamil)

For those unfamiliar with vanilla transformer architecture, you can refer to foundational guides on the topic for a basic overview.

Key concepts of LLaMA include:

### Pre-normalization Using RMSNorm:

LLaMA employs RMSNorm for normalizing the input of each transformer sub-layer. This method, inspired by GPT-3, optimizes computational costs associated with Layer Normalization. RMSNorm provides similar performance to LayerNorm but reduces running time by approximately 7% to 64%.

<img src="https://cdn-images-1.medium.com/max/3604/1*9FA6P93WhRuWFXxVlPG3LA.png" alt="Root Mean Square Layer Normalization Paper" style="width: 50%;">

It achieves this by emphasizing re-scaling invariance and regulating summed inputs based on the root mean square (RMS) statistic, simplifying LayerNorm by removing the mean statistic.

### SwiGLU Activation Function:

LLaMA introduces the SwiGLU activation function, drawing inspiration from PaLM. SwiGLU extends the Swish activation function and involves a custom layer with a dense network to split and multiply input activations.

<img src="https://cdn-images-1.medium.com/max/13536/1*N3dwnqNUD0TdwPYO0NlhYg.png" alt="SwiGLU: GLU Variants Improve Transformer" style="width: 50%;">

The goal is to enhance the expressive power of the model through a more sophisticated activation function.

### Rotary Embeddings (RoPE):

Rotary Embeddings, or RoPE, are used for position embedding in LLaMA. They encode absolute positional information using a rotation matrix and include explicit relative position dependency in self-attention formulations. RoPE offers scalability to various sequence lengths and decaying inter-token dependency with increasing relative distances.

In addition to these concepts, the LLaMA paper utilizes the AdamW optimizer, efficient multi-head attention operators, and manually implemented backward functions to optimize computation.

Special acknowledgment to Anush Kumar for detailed explanations of these LLaMA aspects.

## Setting the Stage

The project utilizes several standard Python libraries:

```python
# PyTorch for implementing LLM
import torch

# Neural network modules and functions from PyTorch
from torch import nn
from torch.nn import functional as F

# NumPy for numerical operations
import numpy as np

# Matplotlib for plotting loss
from matplotlib import pyplot as plt

# Time module for tracking execution time
import time

# Pandas for data manipulation and analysis
import pandas as pd

# urllib for handling URL requests
import urllib.request
```

A configuration object is used to manage model parameters:
```python
# Configuration object for model parameters
MASTER_CONFIG = {
    # Parameters added during implementation
}
```

## Data Preprocessing

While the original LLaMA was trained on 1.4 trillion tokens, this implementation uses a scaled-down approach suitable for smaller projects. We use the TinyShakespeare dataset, which contains approximately 1 million characters.

```python
# Download the dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_name = "tinyshakespeare.txt"
urllib.request.urlretrieve(url, file_name)
```

The vocabulary is determined by the unique characters in the dataset:
```python
lines = open("tinyshakespeare.txt", 'r').read()
vocab = sorted(list(set(lines)))
print('Vocabulary Size:', len(vocab))

# Mapping integers to characters (itos) and vice versa (stoi)
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}
```

A basic character-level tokenizer is implemented for simplicity:
```python
def encode(s):
    return [stoi[ch] for ch in s]

def decode(l):
    return ''.join([itos[i] for i in l])
```

The dataset is converted into a PyTorch tensor:
```python
dataset = torch.tensor(encode(lines), dtype=torch.int8)
```

A batching function handles the data splitting for training and evaluation:
```python
def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]

    batch_data = train
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = test

    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()
    return x, y

MASTER_CONFIG.update({
    'batch_size': 8,
    'context_window': 16
})
```

## Evaluation Strategy

Continuous evaluation is performed during training to monitor performance:
```python
@torch.no_grad()
def evaluate_loss(model, config=MASTER_CONFIG):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out
```

## Setting Up a Base Neural Network Model

We start with a basic neural network and iteratively improve it using LLaMA techniques.

```python
class SimpleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        logits = self.linear(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        else:
            return logits

MASTER_CONFIG.update({'d_model': 128})
model = SimpleModel(MASTER_CONFIG)
```

## Replicating LLaMA Architecture

The implementation incorporates three primary modifications:

### 1. RMSNorm for pre-normalization:

```python
class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw
```

### 2. Rotary Embeddings (RoPE):

RoPE rotates embeddings to encode positional information.

```python
def get_rotary_matrix(context_window, embedding_dim):
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    for position in range(context_window):
        for i in range(embedding_dim // 2):
            theta = 10000. ** (-2. * (i - 1) / embedding_dim)
            m_theta = position * theta
            R[position, 2 * i, 2 * i] = np.cos(m_theta)
            R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
    return R
```

### 3. SwiGLU Activation Function:

```python
class SwiGLU(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)
        return out
```

## Final LLaMA Implementation

The final model combines these components into a multi-layer architecture.

```python
class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )
        self.ffn = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.llama_blocks(x)
        logits = self.ffn(x)
        if targets is None:
            return logits
        else:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
```

## Experimenting with Hyperparameters

Hyperparameter tuning is essential for optimal performance. While the original paper used Cosine Annealing, different schedules can be tested:

```python
MASTER_CONFIG.update({"epochs": 1000})
llama_optimizer = torch.optim.Adam(
    llama.parameters(),
    betas=(.9, .95),
    weight_decay=.1,
    eps=1e-9,
    lr=1e-3
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(llama_optimizer, 300, eta_min=1e-5)
```

## Saving Your Language Model (LLM)

The model can be saved entirely or as a state dictionary:

```python
# Save the entire model
torch.save(llama, 'llama_model.pth')

# Save model parameters
torch.save(llama.state_dict(), 'llama_model_params.pth')
```

## Conclusion

This project demonstrates the step-by-step implementation of the LLaMA architecture. By scaling down the parameters and using a manageable dataset, we can observe how specific architectural choices like RMSNorm, RoPE, and SwiGLU contribute to model performance. This implementation serves as a foundation for further experimentation and fine-tuning for specific use cases.

## Maintainer

**Anarv Singh**
Software Engineer | Backend Engineer

Anarv is a Software Engineer with over 5 years of experience specializing in backend development, distributed systems, and scalable architecture. He has extensive experience building production-scale applications using Java, Python, and cloud-native technologies.

- **GitHub:** [github.com/Anarvsingh](https://github.com/Anarvsingh)
- **LinkedIn:** [linkedin.com/in/anarv-s-91811a173/](https://www.linkedin.com/in/anarv-s-91811a173/)
- **Email:** anarvsingh@gmail.com