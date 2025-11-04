# Voxel-Based Deep Learning

Large-scale voxel-based neural network experiments for 3D point cloud processing.

## Purpose

Comprehensive training pipelines for:
- Voxel-based 3D CNN models
- Point-based models
- Large dataset handling
- Extensive logging and checkpointing
- Reproducible experiments

## Files

### Main Training Scripts (Large Files!)

- **`main_voxel.py`** (6,491 lines) - Voxel model training pipeline
- **`main_points.py`** (6,544 lines) - Point-based model training pipeline
- **`dataset_voxel.py`** (9,293 lines) - Voxel dataset loader
- **`dataset_points.py`** (7,937 lines) - Point dataset loader

### Utilities

- `metrics.py` - Evaluation metrics
- `utils_voxel.py` - Voxel utility functions
- `scipy_tests.py` - Rotation and transformation tests

### Logs

- `logs/log_100.pdf` - Example training log (100 epochs)

## Why Are Files So Large?

These files contain **complete experimental pipelines** in single scripts:

- Data loading and preprocessing
- Model architecture definitions
- Training loops with all logic
- Validation and testing
- Visualization code
- Metrics computation
- Hyperparameter configurations
- Logging and checkpointing

This is **intentional for ML research**:
- Keeps experiment context together
- Easier to reproduce results
- Self-contained experiments
- Common in academic research

**Navigation**: See table of contents at top of each large file!

## Quick Start

### Training Voxel Model

```bash
python main_voxel.py \
    --data_path ./data \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --voxel_size 32
```

### Training Point Model

```bash
python main_points.py \
    --data_path ./data \
    --batch_size 32 \
    --epochs 100 \
    --num_points 2048
```

## File Structure (Explained)

### main_voxel.py - Table of Contents

Approximate line ranges:
- **Lines 1-500**: Imports, config, data loading
- **Lines 501-1500**: Voxel model architecture
- **Lines 1501-3000**: Training loop
- **Lines 3001-4000**: Validation logic
- **Lines 4001-5000**: Visualization functions
- **Lines 5001-6000**: Metrics and logging
- **Lines 6001-6491**: Main execution, argument parsing

Key functions:
- `voxelize()` - Convert points to voxel grid
- `VoxelCNN()` - Model class
- `train_epoch()` - Training loop
- `validate()` - Validation logic
- `main()` - Entry point

### dataset_voxel.py - Table of Contents

Approximate line ranges:
- **Lines 1-1000**: Dataset class definition
- **Lines 1001-3000**: Voxelization methods
- **Lines 3001-5000**: Augmentation functions
- **Lines 5001-7000**: Preprocessing utilities
- **Lines 7001-9293**: I/O and caching

## Dependencies

```bash
pip install torch>=2.0.0
pip install numpy>=1.24.0
pip install scipy>=1.10.0
pip install matplotlib>=3.7.0
```

## Voxel Representation

### What is Voxelization?

Convert point cloud to 3D grid:

```python
# Point cloud: (N, 3) - variable size
points = np.random.rand(5000, 3) * 10  # In [0, 10]

# Voxel grid: (32, 32, 32) - fixed size
voxel_grid = voxelize(points, voxel_size=32, bounds=[0, 10])

# Each voxel: occupied (1) or empty (0)
# Or: point density, colors, features
```

**Advantages:**
- Fixed-size input for CNNs
- 3D convolutions work naturally
- Captures spatial structure

**Disadvantages:**
- Resolution limited by voxel size
- Memory grows as O(n³)
- Loses fine details

## Model Architecture

### Voxel CNN

```python
class VoxelCNN(nn.Module):
    def __init__(self, voxel_size=32, num_classes=40):
        super().__init__()
        # 3D Convolutions
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)

        # Pooling reduces spatial dimensions
        self.pool = nn.MaxPool3d(2)

        # Classifier
        self.fc = nn.Linear(128 * (voxel_size//8)**3, num_classes)

    def forward(self, x):
        # x: (batch, 1, voxel_size, voxel_size, voxel_size)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

## Training Configuration

### Hyperparameters

```python
config = {
    # Data
    'voxel_size': 32,           # Grid resolution
    'batch_size': 32,
    'num_workers': 4,

    # Training
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'optimizer': 'Adam',

    # Augmentation
    'rotation': True,           # Random rotation
    'jitter': 0.01,             # Point jitter
    'dropout': 0.3,             # Voxel dropout

    # Checkpointing
    'save_freq': 10,            # Save every N epochs
    'log_freq': 100             # Log every N batches
}
```

## Data Organization

### Expected Directory Structure

```
data/
├── train/
│   ├── class1/
│   │   ├── sample1.npy
│   │   ├── sample2.npy
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Each `.npy` file contains point cloud: `(N, 3)` numpy array.

### Dataset Creation

```python
from python.pytorch_voxel import dataset_voxel

dataset = dataset_voxel.VoxelDataset(
    root='./data/train',
    voxel_size=32,
    transform=augmentation,
    cache=True          # Cache voxelized data
)
```

## Memory Management

### Voxel Memory Usage

- 32³ voxels: ~128 KB per sample
- 64³ voxels: ~1 MB per sample
- 128³ voxels: ~8 MB per sample

**Tips:**
- Use sparse voxels for large grids
- Cache preprocessed voxels
- Use float16 if possible
- Stream from disk for huge datasets

## Checkpointing

Models and training state automatically saved:

```
model_checkpoint/
├── epoch_010.pth
├── epoch_020.pth
├── ...
├── best_model.pth
└── last_model.pth
```

Resume training:
```bash
python main_voxel.py \
    --resume model_checkpoint/epoch_050.pth \
    --epochs 100
```

## Visualization

Training includes automatic visualization:
- Loss curves
- Accuracy plots
- Sample predictions
- Confusion matrices

Logs saved to `logs/` directory.

## Performance Tips

### Speed Up Training

1. **Use GPU**:
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

2. **Enable cuDNN benchmarking**:
   ```python
   torch.backends.cudnn.benchmark = True
   ```

3. **Increase num_workers**:
   ```python
   DataLoader(..., num_workers=4, pin_memory=True)
   ```

4. **Mixed precision**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

### Reduce Memory Usage

1. **Smaller voxel size** (32 instead of 64)
2. **Smaller batch size**
3. **Gradient accumulation**
4. **Clear cache**: `torch.cuda.empty_cache()`

## Common Issues

**CUDA out of memory:**
```python
# Reduce batch_size
batch_size = 16  # or 8

# Or reduce voxel_size
voxel_size = 24  # instead of 32
```

**Training too slow:**
```python
# Check GPU is being used
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Increase num_workers
num_workers = 4
```

**Dataset not found:**
```bash
# Check data path
ls ./data/train

# Verify .npy files exist
find ./data -name "*.npy" | head
```

## Evaluation Metrics

Computed automatically:
- Overall accuracy
- Per-class accuracy
- Confusion matrix
- Top-5 accuracy
- F1 score

See `metrics.py` for implementations.

## Learning Objectives

- Voxel representations for 3D data
- 3D Convolutional Neural Networks
- Large-scale dataset handling
- Training pipeline design
- Experiment reproducibility
- Memory optimization

## Comparison: Points vs Voxels

| Aspect | Point-Based | Voxel-Based |
|--------|-------------|-------------|
| **Input size** | Variable | Fixed |
| **Memory** | O(n) points | O(k³) voxels |
| **Operations** | Special (PointNet) | Standard CNNs |
| **Detail** | High | Limited by resolution |
| **Speed** | Fast | Slower (3D conv) |

## References

- [3D ShapeNets](https://3dshapenets.cs.princeton.edu/)
- [VoxNet Paper](https://www.ri.cmu.edu/pub_files/2015/9/voxnet_maturana_scherer_iros15.pdf)
- [ModelNet](https://modelnet.cs.princeton.edu/)
- [Point Cloud Deep Learning Survey](https://arxiv.org/abs/1912.12033)

## Notes

- Large file sizes are intentional for research code
- See table of contents in files for navigation
- Training logs saved automatically
- Checkpoints saved every N epochs
- Use `main_voxel.py --help` for all options
