# MDCDGCN: Multi-Scale Deformable Convolutional Graph Convolutional Network for Depression Detection

MDCDGCN is a deep learning model designed for the detection of Major Depressive Disorder (MDD) using EEG data. It incorporates four innovative modules that work together to dynamically learn EEG features at both temporal and spatial levels:

## Architecture Overview

### 1. **Adaptive Graph Convolution Transformation Module (AGCTM)**

AGCTM dynamically learns the spatial dependencies between EEG electrodes during training. This allows the model to adaptively adjust its graph structure for different EEG data. It also learns temporal features through a series of 1D convolutions, batch normalization, and the Mish activation function.

- **Key Operation**: Dynamic graph structure learning with adjacency matrix optimization.
- **Goal**: Capture the evolving functional connectivity between EEG channels over time.

### 2. **Dynamical Feature-Fusion Graph Convolutional Module (DFGCM)**

DFGCM refines the learned spatial features from the AGCTM by dynamically optimizing the adjacency matrix and applying graph convolutions. It also uses residual feature fusion and depthwise convolutions to capture both short-term and long-term temporal dependencies in EEG signals.

- **Key Operation**: Feature fusion with residual connections and depthwise convolutions for long-term temporal modeling.
- **Goal**: Integrate local temporal fluctuations and global long-term trends in EEG signals.

### 3. **Multi-Scale Deformable Convolution Module (MDCM)**

MDCM uses deformable convolutions with multi-scale receptive fields to capture dynamic temporal patterns in EEG signals. The module adjusts its receptive field using learnable offsets, allowing it to capture both short-term fluctuations and long-term trends in the EEG data.

- **Key Operation**: Multi-scale deformable convolutions with learnable offsets.
- **Goal**: Capture both localized fluctuations and long-term trends in EEG signals for better MDD classification.

### 4. **Adaptive Feature Preprocessing Module (AFPM)**

AFPM integrates both local and global features from EEG signals. It uses two branches—one focusing on local feature extraction using depthwise convolutions and the other performing global feature aggregation using the normalized Laplacian matrix. The outputs from both branches are fused to form a comprehensive representation of EEG activity.

- **Key Operation**: Local feature extraction with depthwise convolutions and global feature aggregation using the Laplacian matrix.
- **Goal**: Combine local and global EEG features to enhance the detection of MDD.
