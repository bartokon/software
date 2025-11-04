# PyTorch Point Cloud Models

Deep learning models and training scripts for point cloud processing using PyTorch and PyTorch Geometric.

## Purpose

Collection of neural network architectures for:
- Direct point cloud processing (PointNet)
- Graph neural networks for point clouds
- Custom model architectures
- Training pipelines
- Dataset loaders

## Files

### Models
- `pointnet_model.py` - PointNet architecture
- `mymodel_0.py`, `mymodel_1.py`, `mymodel_2.py` - Custom architectures
- `model.py`, `model_pn.py` - Base models
- `histogram_model.py` - Histogram-based features

### Datasets
- `dataset.py` - Main dataset class
- `dataset_pytorch.py` - PyTorch dataset wrapper
- `histogram_dataset.py` - Histogram feature dataset

### Training
- `main.py` - Basic training script
- `main_gcn.py` - Graph Convolutional Network training
- `main_gcn_ConvNet.py` - ConvNet training
- `main_gcn_pointnet.py` - PointNet training

### Utilities
- `rotate.py` - Point cloud rotation augmentation
- `arrow_3d.py` - 3D visualization
- `mesh_io.py` - Mesh file I/O
- `point_information.py` - Point cloud statistics

### Scripts
- `clean.sh` - Clean generated files
- `copy.sh` - Copy utilities

## Dependencies

```bash
pip install torch>=2.0.0
pip install torch-geometric>=2.3.0
pip install numpy matplotlib scipy
```

## Usage

### Training PointNet

```bash
python main_gcn_pointnet.py \
    --data_path ./data \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001
```

### Using PointNet Model

```python
from python.pytorch import pointnet_model
import torch

# Create model
model = pointnet_model.PointNet(num_classes=10)

# Input: (batch, num_points, 3)
points = torch.rand(32, 1024, 3)

# Forward pass
output = model(points)  # (batch, num_classes)
```

### Custom Training Loop

```python
from python.pytorch import dataset, model

# Load data
train_dataset = dataset.PointCloudDataset('./data/train')
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# Create model
net = model.CustomModel()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    for batch_points, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = net(batch_points)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
```

## Models

### PointNet

Directly processes unordered point sets:

```python
PointNet(
    num_points=1024,    # Number of input points
    num_classes=40,     # Classification classes
    dropout=0.3         # Dropout rate
)
```

**Architecture:**
1. Shared MLPs on each point
2. Max pooling for permutation invariance
3. Global feature extraction
4. Classification head

**Key Features:**
- Permutation invariant
- Rotation augmentation
- Handles variable input size
- Fast inference

### Graph Neural Networks

Uses PyTorch Geometric for graph convolutions:

```python
from python.pytorch import model

gcn = model.GCNModel(
    input_dim=3,        # Point coordinates
    hidden_dim=64,
    output_dim=10
)
```

### Custom Models

Experimental architectures in `mymodel_*.py`:
- Multi-scale features
- Attention mechanisms
- Hierarchical processing

## Datasets

### Point Cloud Dataset

```python
from python.pytorch import dataset

ds = dataset.PointCloudDataset(
    root='./data',
    split='train',
    num_points=1024,    # Sample to this many points
    transform=None
)

points, label = ds[0]
print(points.shape)  # (1024, 3)
```

### Histogram Dataset

Uses histogram features instead of raw points:

```python
from python.pytorch import histogram_dataset

ds = histogram_dataset.HistogramDataset(
    root='./data',
    bins=32             # Histogram bins per dimension
)
```

## Data Augmentation

### Random Rotation

```python
from python.pytorch import rotate

def augment(points):
    # Random rotation around Z axis
    angle = np.random.uniform(0, 2*np.pi)
    rotated = rotate.rotate_point_cloud_z(points, angle)
    return rotated
```

### Random Jitter

```python
def add_jitter(points, sigma=0.01):
    noise = np.random.normal(0, sigma, points.shape)
    return points + noise
```

### Random Dropout

```python
def random_dropout(points, max_dropout=0.5):
    keep_ratio = np.random.uniform(0.5, 1.0)
    num_keep = int(len(points) * keep_ratio)
    indices = np.random.choice(len(points), num_keep, replace=False)
    return points[indices]
```

## Training Parameters

Common hyperparameters:

```python
config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'optimizer': 'Adam',
    'scheduler': 'StepLR',
    'weight_decay': 1e-4,
    'dropout': 0.3
}
```

## Evaluation

### Metrics

```python
from python.pytorch import point_information

# Accuracy
accuracy = (predictions == labels).float().mean()

# Per-class accuracy
per_class_acc = point_information.per_class_accuracy(
    predictions, labels, num_classes=40
)

# Confusion matrix
cm = point_information.confusion_matrix(predictions, labels)
```

## Visualization

### Plot Point Cloud

```python
from python.pytorch import arrow_3d

arrow_3d.plot_point_cloud(
    points,
    colors=labels,      # Color by label
    title="Point Cloud"
)
```

### Plot Training Curves

```python
import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

## GPU Support

```python
import torch

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model and data to GPU
model = model.to(device)
points = points.to(device)

# For AMD GPUs (ROCm)
# Uncomment if needed:
# import os
# os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
```

## Common Issues

**Out of memory:**
- Reduce batch_size
- Reduce num_points
- Use gradient accumulation

**Slow training:**
- Ensure using GPU (`torch.cuda.is_available()`)
- Use DataLoader with num_workers>0
- Enable cuDNN benchmarking: `torch.backends.cudnn.benchmark = True`

**NaN loss:**
- Reduce learning rate
- Check for inf/nan in data
- Add gradient clipping

## Learning Objectives

- Deep learning on point clouds
- PointNet architecture details
- Graph neural networks
- PyTorch training loops
- Data augmentation techniques
- GPU acceleration

## References

- [PointNet Paper](https://arxiv.org/abs/1612.00593)
- [PointNet++ Paper](https://arxiv.org/abs/1706.02413)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [ModelNet40 Dataset](https://modelnet.cs.princeton.edu/)
