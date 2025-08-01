# Neural Network Precision Comparison Framework

A comprehensive PyTorch-based framework for comparing the performance of neural networks across different numerical precisions (FP2, FP3, FP4, FP6, FP8, FP32) with various activation functions and model architectures.

## Overview

This project implements and compares neural network models using different floating-point precisions to analyze the trade-offs between model accuracy. The framework supports multiple architectures, activation functions, and datasets to provide comprehensive benchmarking capabilities.

## Features

- **Multiple Precision Support**: FP2, FP3, FP4, FP6, FP8, and FP32
- **6 Neural Network Architectures**: CNN, GCN, MLP-Mixer, MobileNetV2, ResNet18, Vision Transformer
- **9 Activation Functions**: ReLU, Sigmoid, Tanh, GELU, Leaky ReLU, PReLU, Swish, Softplus
- **Precision-Specific Softmax**: Independent softmax precision control for output layers
- **2 Datasets**: CIFAR-10, CIFAR-100
- **Comprehensive Metrics**: Accuracy, loss, training time, memory usage
- **Automated Visualization**: Performance plots and comparisons
- **Memory Tracking**: Peak memory usage and estimated footprint analysis

## Project Structure

```
├── config.yaml                    # Configuration file
├── main.py                        # Main experiment runner
├── utils.py                       # Utility functions and plotting
├── datasets/                      # Dataset loaders
│   ├── cifar10.py
│   └── cifar100.py
├── fp2_models/                    # FP2 precision models
│   ├── __init__.py
│   ├── act.py                     # FP2 activation functions
│   ├── fp2_cnn.py
│   ├── fp2_gcn.py
│   ├── fp2_mlpmixer.py
│   ├── fp2_mobilenetv2.py
│   ├── fp2_resnet18.py
│   └── fp2_vit.py
├── fp3_models/                    # FP3 precision models (similar structure)
├── fp4_models/                    # FP4 precision models (similar structure)
├── fp6_models/                    # FP6 precision models (similar structure)
├── fp8_models/                    # FP8 precision models (similar structure)
├── fp32_models/                   # FP32 precision models
│   ├── fp32_cnn.py
│   ├── fp32_gcn.py
│   ├── fp32_mlpmixer.py
│   ├── fp32_mobilenetv2.py
│   ├── fp32_resnet18.py
│   └── fp32_vit.py
├── softmax_layers/                # Precision-specific softmax implementations
│   └── softmax_layers.py
├── plots/                         # Generated visualization plots
└── final_results.csv              # Experiment results
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision matplotlib pandas seaborn pyyaml psutil
```

### Configuration

Edit `config.yaml` to specify your experiment parameters:

## Configuration Options

### Model Selection
Choose from: `cnn`, `gcn`, `mlpmixer`, `mobilenet`, `resnet`, `vit`

### Activation Functions
Choose from: `sigmoid`, `tanh`, `relu`, `leaky_relu`, `gelu`, `prelu`, `swish`, `softplus`

### Datasets
Choose from: `cifar10`, `cifar100`

### Precisions
Choose from: `fp2`, `fp3`, `fp4`, `fp6`, `fp8`, `fp32`

### Running Experiments

```bash
python main.py
```

This will:
1. Load the configuration
2. Run experiments for all specified combinations
3. Log results to `final_results.csv`
4. Generate visualization plots in the `plots/` directory

## Model Architectures

### 1. **Convolutional Neural Network (CNN)**
- 4 convolutional blocks with batch normalization
- Progressive channel expansion: 64 → 128 → 256 → 512
- Adaptive average pooling and fully connected classifier

### 2. **Graph Convolutional Network (GCN)**
- Converts images to graph representations using spatial patches
- 3 graph convolutional layers with normalized adjacency matrix
- Grid-based connectivity for spatial relationships

### 3. **MLP-Mixer**
- Simple multi-layer perceptron architecture
- Flattened input with hidden layers
- Configurable hidden dimensions

### 4. **MobileNetV2**
- Based on torchvision's MobileNetV2
- Depth-wise separable convolutions
- Inverted residual blocks

### 5. **ResNet18**
- Based on torchvision's ResNet18
- Residual connections and skip connections
- Batch normalization throughout

### 6. **Vision Transformer (ViT)**
- Patch-based image processing
- Multi-head self-attention mechanism
- Positional embeddings and classification token

## Precision Implementation

### Low-Precision Quantization

The framework implements custom quantization using Straight-Through Estimator (STE):

```python
class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, vals):
        # Quantize to nearest values
        idx = torch.searchsorted(vals, x, right=False)
        idx = torch.clamp(idx, 1, len(vals) - 1)
        # ... quantization logic
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # Straight-through gradients
```

### Learnable Quantization Values

Each precision level uses learnable quantization values:
- **FP2**: 4 learnable values (e.g., [-1.0, 0.0, 1.0, 4.0])
- **FP3**: 8 learnable values
- **FP4**: 16 learnable values
- And so on...

## Metrics and Analysis

### Tracked Metrics
- **Training/Test Accuracy**
- **Training/Test Loss**
- **Epoch Time**
- **Peak Memory Usage**
- **Estimated Memory Footprint**

## Visualization

Automated plot generation includes:
- **Accuracy vs Epoch**
- **Loss vs Epoch**
- **Memory Usage**
- **Training Time**
- **Memory Comparison**

## Customization

### Adding New Models
1. Create model file in appropriate precision directory
2. Implement the model class with required activation parameter
3. Add to model mapping in `utils.py`

### Adding New Activation Functions
1. Implement quantized version in precision-specific `act.py`
2. Add to activation mapping in model constructors

### Adding New Datasets
1. Create dataset loader in `datasets/` directory
2. Return train_loader, test_loader, input_shape, num_classes
3. Add to dataset mapping in `utils.py`

## Results Format

Results are saved to `final_results.csv` with columns:
- model, activation, dataset, precision, epoch
- train_acc, train_loss, test_acc, test_loss
- time, memory, estimated_memory

## License

This project is open source and available under the MIT License.