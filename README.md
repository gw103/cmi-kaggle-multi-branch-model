# CMI Kaggle Multi-Branch Model

A sophisticated multi-sensor fusion approach for detecting Body-Focused Repetitive Behaviors (BFRBs) using data from the Helios wrist-worn device. This project implements a robust multi-branch neural network architecture designed to distinguish between BFRB-like behaviors (hair pulling, skin picking) and non-BFRB everyday gestures (adjusting glasses, drinking) using IMU, thermal, and ToF sensor data.

## Problem Statement

Body-Focused Repetitive Behaviors (BFRBs) such as hair pulling (trichotillomania), skin picking (excoriation), and nail biting are self-directed habits that can cause physical harm and psychosocial challenges. These behaviors are commonly associated with anxiety disorders and obsessive-compulsive disorder (OCD), making them important indicators of mental health challenges.

The Child Mind Institute's Helios device collects multi-modal sensor data to detect these behaviors using IMU sensors (accelerometer and gyroscope data with 7 channels), 5 thermopile sensors measuring body heat, and 5 Time-of-Flight sensors with 64 distance measurements each arranged in an 8×8 grid. The key constraint is that approximately half of the test sequences contain only IMU data, while the other half includes all sensors, testing whether additional thermal and ToF sensors provide significant value over IMU-only detection.

The classification task involves both binary classification (BFRB-like vs non-BFRB-like gestures) and multi-class classification across 8 BFRB-like gestures and 10 non-BFRB-like gestures.

## Key Innovations

**Intelligent ToF Data Handling**: The original ToF data contains -1 values indicating sensor failures or out-of-range measurements. Instead of treating these as missing data, we replace -1 values with large positive values (512) to represent "far distance." This approach provides meaningful information for BFRB detection, as failed ToF readings likely indicate objects are beyond the sensor's detection range, which is crucial for distinguishing behaviors like hand positioning during hair pulling versus glasses adjustment.

**Masked Attention for Sensor Fusion**: We implement a multi-head attention mechanism with masking capabilities that dynamically weights and fuses features from different sensor modalities. This 3-layer transformer-style attention with residual connections enables the model to gracefully handle sensor failures by learning to rely on available sensors, providing robustness for real-world deployment scenarios.

**Dual Model Strategy with Smart Inference**: We develop specialized models for different data availability scenarios. The full multi-sensor model processes all available sensor data (IMU + Thermal + ToF), while the IMU-only model is trained exclusively on IMU data for sequences without other sensors. During inference, the system automatically detects missing sensor data and routes to the appropriate model, ensuring optimal performance for both complete and incomplete sensor scenarios.

**1D CNN Backbone with Feature Pyramid Network**: Our architecture uses 1D Convolutional Neural Networks for temporal feature extraction, integrated with a Feature Pyramid Network (FPN) for multi-scale feature representation. This approach captures both local temporal patterns and global sequence characteristics through residual blocks with skip connections for improved gradient flow.

## Model Architecture

The multi-sensor model processes input from IMU (20D), Thermal (5D), and ToF (5×64D) sensors through separate sensor branches. Each branch uses 1D CNN with residual blocks and FPN for feature extraction, followed by GRU layers for temporal modeling. The features are then fused through a masked multi-head attention mechanism with 3 layers and residual connections, ultimately producing classification outputs for 18 gesture classes.

The IMU-only model takes IMU data with 28 engineered features through a feature MLP (128 → 256 → 256 with GELU activation), followed by multi-scale temporal convolutions with FPN. The architecture includes pyramid pooling with adaptive average and max pooling at 1, 2, and 4 scales, leading to final classification for 18 gesture classes.

## Technical Implementation

**Feature Engineering**: The IMU features include 7 raw sensors (acc_x, acc_y, acc_z, rot_x, rot_y, rot_z, rot_w), magnitude features (acc_mag, rot_angle), derivatives (acc_mag_jerk, rot_angle_vel), gravity-removed linear acceleration components, and angular dynamics including angular velocity and distance calculations. The IMU-only model uses additional engineered features (28 total) compared to the full model (20 total) to compensate for the lack of additional sensor modalities.

**ToF Data Preprocessing**: Failed readings are replaced with 512 to indicate "far distance," while NaN values are handled with 1000. Attention masks are created for failed sensors using the condition `~(np.isnan(arr).all(axis=1) | (arr == -1).all(axis=1))`, and masking is applied in the forward pass by setting masked values to 0.

**Training Strategy**: Data augmentation includes mixup interpolation between samples of the same gesture class, Gaussian noise injection, random amplitude scaling, and temporal sequence reversal. Regularization techniques include label smoothing (0.1 factor), gradient clipping (max norm 1.0), progressive dropout rates (0.3-0.8), and branch masking with random sensor dropout during training. Optimization uses AdamW with weight decay, OneCycleLR with cosine annealing, learning rates of 1e-4 for multi-sensor and 1e-3 for IMU-only models, with batch sizes of 64 for training and 128 for validation.

## Performance Metrics

The models are evaluated using the competition's official metrics: Binary F1 Score for BFRB-like vs non-BFRB-like gesture classification, Macro F1 Score averaged across all gesture classes (with non-target gestures collapsed into single class), and Combined Score as the average of Binary F1 and Macro F1. Cross-validation uses 5-fold stratified grouping with subject-based splitting, early stopping with patience of 15 epochs, best model selection based on combined score, and evaluation through an API that requires single sequence inference.

## Inference Strategy

The inference pipeline implements intelligent model selection with automatic sensor availability detection. The system first checks for missing values in thermal and ToF sensor data. If both sensor types are available, it routes to the full multi-sensor model ensemble (3 models). If any sensor data is missing, it automatically switches to the IMU-only model ensemble (10 models). Both models output the same gesture classification format, ensuring seamless prediction regardless of sensor availability.

The system uses weighted ensemble averaging for final predictions, with the IMU-only ensemble using 10 models and the full model ensemble using 3 models. Feature engineering is applied consistently across both models, with the IMU-only model using additional engineered features (28 total) compared to the full model (20 total) to compensate for the lack of additional sensor modalities.

## Key Advantages

The approach provides robustness through graceful handling of sensor failures via masked attention, flexibility through dual model architecture for different data availability scenarios, efficiency through 1D CNN backbone optimized for temporal data, scalability through Feature Pyramid Network enabling multi-scale feature extraction, and strong generalization through extensive data augmentation and regularization techniques.

## Research Contributions

This work contributes novel ToF data interpretation by treating failed readings as "far distance" rather than missing data, providing meaningful information for BFRB detection. We introduce masked multi-modal fusion with attention-based sensor failure handling for robust wearable device applications. The dual-model architecture addresses real-world deployment constraints for different sensor availability scenarios. Our 1D CNN + FPN approach provides effective temporal feature extraction with multi-scale representation for gesture recognition, and establishes a framework for determining sensor value in mental health monitoring devices.

## Clinical Impact

This work directly contributes to the development of better tools for detecting and treating BFRBs by enabling early detection for timely intervention, providing objective measures of behavior frequency and patterns for treatment monitoring, informing decisions about sensor selection for cost-effective wearable devices, and strengthening tools available for treating anxiety disorders and OCD.

## Competition Details

The competition is organized by the Child Mind Institute using Helios wrist-worn device sensor data. It involves 8 BFRB-like gestures and 10 non-BFRB-like gestures, evaluated using Binary F1 + Macro F1 (equally weighted). The test set contains approximately 3,500 sequences with 50% IMU-only and 50% full sensor data, running from May 2025 to September 2025.

## File Structure

```
cmi-kaggle-multi-branch-model/
├── README.md                    # This file
├── Full_model_no_gru.ipynb     # Multi-sensor model implementation
└── imu_only.ipynb              # IMU-only model implementation
```

## Usage

The multi-sensor model can be initialized with `MultiSensorClassifier(imu_input_dim=len(imu_cols), thm_input_dim=5, tof_input_dim=64, hidden_dim=256, num_classes=18)` and trained using cross-validation. The IMU-only model uses `ImprovedCNNIMUModel(input_dim=len(imu_cols), num_classes=len(gestures), dropout_rate=0.25)` with specialized augmentation through `IMUDataset(df, imu_cols, label_map, augment=True)`.

The smart inference pipeline automatically detects sensor availability and routes to the appropriate model. If thermal and ToF data are available, it uses the full multi-sensor model; otherwise, it uses the IMU-only model, ensuring optimal performance for each scenario.

## Future Improvements

Potential enhancements include ensemble methods combining predictions from both models for improved accuracy, attention visualization to analyze which sensors contribute most to BFRB detection, online learning to adapt to new sensor failure patterns and user-specific behaviors, efficiency optimization through model compression for edge deployment on wearable devices, clinical validation through real-world testing with patients experiencing BFRBs, and sensor cost-benefit analysis to quantify the value of additional sensors for clinical applications.

---

*This implementation represents a comprehensive approach to multi-modal sensor fusion for BFRB detection, with particular emphasis on robustness, real-world applicability, and clinical relevance for mental health monitoring devices.*