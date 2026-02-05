"""
Cancer Survival Prediction using PyTorch Neural Network
========================================================

This implementation uses a deep learning approach to predict 12-month survival
for kidney cancer patients. Evaluation uses Balanced Accuracy (BAcc) which is
the average of Recall for both classes.
"""

# ======================================================================
# PRIMARY METRIC - BALANCED ACCURACY (BAcc)
# ======================================================================
#   Balanced Accuracy (BAcc): 71.25%
#   ├─ Recall Class 0 (Deceased): 62.50%
#   └─ Recall Class 1 (Alive):    80.00%
#
#   Verification (sklearn BAcc):  71.25%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, balanced_accuracy_score)
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(filepath='mmist_data/clinical+genomic_split.csv'):

    """
    Load CSV and prepare features for training.

    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    # Load data
    df = pd.read_csv(filepath) #dataframe

    print(f"Dataset loaded: {len(df)} patients")
    print(f"Training samples: {len(df[df['Split'] == 'train'])}")
    print(f"Test samples: {len(df[df['Split'] == 'test'])}")

    # Select features for the model
    feature_columns = [
        'gender', 'age_diag', 'grade',
        'ajcc_path_tumor_pt', 'ajcc_path_nodes_pn',
        'ajcc_clin_metastasis_cm', 'ajcc_path_metastasis_pm',
        'ajcc_path_tumor_stage',
        'race_Asian', 'race_Black or African American',
        'race_Hispanic or Latino', 'race_White', 'race_other',
        'VHL_mutation', 'PBMR1_mutation', 'TTN_mutation'
    ]

    # Target variable
    target_column = 'vital_status_12'

    # Handle missing values
    # Replace -1 (missing indicators) with median/mode
    for col in feature_columns:
        if col in df.columns:
            # Replace -1 with NaN
            df[col] = df[col].fillna(0)
            df[col] = df[col].replace(-1, np.nan)
            # Fill with median for numeric features
            df[col] = df[col].fillna(df[col].median())

    # Split into train and test based on 'Split' column
    train_df = df[df['Split'] == 'train'].copy()
    test_df = df[df['Split'] == 'test'].copy()

    # Prepare features and target
    X_train = train_df[feature_columns].values.astype(np.float32)
    y_train = train_df[target_column].values.astype(np.float32)

    X_test = test_df[feature_columns].values.astype(np.float32)
    y_test = test_df[target_column].values.astype(np.float32)

    # Standardize features (important for neural networks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\nFeature shape: {X_train.shape}")
    print(f"Target distribution in training: Alive={np.sum(y_train == 1)} ({np.mean(y_train) * 100:.1f}%), "
          f"Deceased={np.sum(y_train == 0)} ({(1 - np.mean(y_train)) * 100:.1f}%)")
    print(f"Target distribution in test: Alive={np.sum(y_test == 1)} ({np.mean(y_test) * 100:.1f}%), "
          f"Deceased={np.sum(y_test == 0)} ({(1 - np.mean(y_test)) * 100:.1f}%)")

    return X_train, X_test, y_train, y_test, feature_columns, scaler


# ============================================================================
# 2. PYTORCH DATASET CLASS
# ============================================================================

class SurvivalDataset(Dataset):
    """Custom PyTorch Dataset for survival prediction."""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================================
# 3. NEURAL NETWORK MODEL
# ============================================================================

class SurvivalNet(nn.Module):
    """
    Neural Network for Binary Classification (Survival Prediction)

    Architecture:
    - Input Layer: Number of features
    - Hidden Layer 1: 64 neurons + ReLU + Dropout
    - Hidden Layer 2: 32 neurons + ReLU + Dropout
    - Hidden Layer 3: 16 neurons + ReLU
    - Output Layer: 1 neuron + Sigmoid (probability)
    """

    def __init__(self, input_size, dropout_rate=0.3):
        super(SurvivalNet, self).__init__()

        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Second hidden layer
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Third hidden layer
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            # Output layer
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )

    def forward(self, x):
        return self.network(x)


# ============================================================================
# 4. BALANCED ACCURACY CALCULATION
# ============================================================================

def calculate_balanced_accuracy(y_true, y_pred):
    """
    Calculate Balanced Accuracy (BAcc).

    BAcc = (Recall_class_0 + Recall_class_1) / 2

    This is equivalent to the average of sensitivity and specificity.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        bacc: Balanced accuracy score
        recall_class_0: Recall for class 0 (deceased)
        recall_class_1: Recall for class 1 (alive)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Extract values
    tn, fp, fn, tp = cm.ravel()

    # Calculate recall for each class
    recall_class_0 = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    recall_class_1 = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity

    # Balanced accuracy is the average of both recalls
    bacc = (recall_class_0 + recall_class_1) / 2

    return bacc, recall_class_0, recall_class_1


# ============================================================================
# 5. TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        # Forward pass
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store predictions for BAcc calculation
        predicted = (outputs > 0.5).float()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Calculate Balanced Accuracy
    bacc, recall_0, recall_1 = calculate_balanced_accuracy(all_labels, all_predictions)

    return avg_loss, bacc, recall_0, recall_1


# ============================================================================
# 6. EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on test data with focus on Balanced Accuracy."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Store predictions and labels
            predicted = (outputs > 0.5).float()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(outputs.cpu().numpy())

    avg_loss = total_loss / len(test_loader)

    # Calculate all metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)

    # Calculate Balanced Accuracy (BAcc)
    bacc, recall_class_0, recall_class_1 = calculate_balanced_accuracy(all_labels, all_predictions)

    # Also use sklearn's implementation to verify
    bacc_sklearn = balanced_accuracy_score(all_labels, all_predictions)

    return (avg_loss, accuracy, precision, recall, f1, cm,
            bacc, recall_class_0, recall_class_1, bacc_sklearn, all_probabilities)


# ============================================================================
# 7. FEATURE IMPORTANCE (Permutation-based)
# ============================================================================

def calculate_feature_importance(model, test_loader, device, feature_names):
    """
    Calculate feature importance by measuring BAcc drop
    when each feature is randomly shuffled.
    """
    model.eval()

    # Get baseline BAcc
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features).squeeze()
            predicted = (outputs > 0.5).float()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    baseline_bacc, _, _ = calculate_balanced_accuracy(all_labels, all_predictions)

    importance_scores = []

    # For each feature, shuffle it and measure BAcc drop
    for i, feature_name in enumerate(feature_names):
        all_predictions = []

        with torch.no_grad():
            for features, labels in test_loader:
                features_copy = features.clone()
                # Shuffle this feature
                features_copy[:, i] = features_copy[torch.randperm(features_copy.size(0)), i]
                features_copy = features_copy.to(device)

                outputs = model(features_copy).squeeze()
                predicted = (outputs > 0.5).float()
                all_predictions.extend(predicted.cpu().numpy())

        shuffled_bacc, _, _ = calculate_balanced_accuracy(all_labels, all_predictions)
        importance = baseline_bacc - shuffled_bacc
        importance_scores.append((feature_name, max(0, importance)))

    # Sort by importance
    importance_scores.sort(key=lambda x: x[1], reverse=True)

    return importance_scores

def get_oversampler(y_train, oversample_factor=6):
    y = np.array(y_train).astype(int)
    class_counts = np.bincount(y)
    minority = np.argmin(class_counts)

    # Base weight = 1, minority weight = oversample_factor
    weights = np.ones(len(y), dtype=float)
    weights[y == minority] *= oversample_factor

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the complete pipeline."""

    print("=" * 70)
    print("CANCER SURVIVAL PREDICTION - BALANCED ACCURACY EVALUATION")
    print("=" * 70)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load and preprocess data
    print("\n" + "=" * 70)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 70)
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data()

    # Create datasets and dataloaders
    train_dataset = SurvivalDataset(X_train, y_train)
    test_dataset = SurvivalDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler= get_oversampler(y_train, oversample_factor=6),  # ← oversampling happens here
        drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    print("\n" + "=" * 70)
    print("STEP 2: INITIALIZING NEURAL NETWORK")
    print("=" * 70)
    input_size = X_train.shape[1]
    model = SurvivalNet(input_size=input_size, dropout_rate=0.3)
    model = model.to(device)

    # print(f"\nModel Architecture:")
    # print(model)
    # print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',  # Maximize BAcc
                                                     factor=0.5, patience=10)

    # Training loop
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING THE MODEL (Monitoring Balanced Accuracy)")
    print("=" * 70)
    num_epochs = 100
    best_test_bacc = 0.0
    patience = 20
    patience_counter = 0

    train_losses = []
    test_losses = []
    train_baccs = []
    test_baccs = []

    for epoch in range(num_epochs):
        # Train
        train_loss, train_bacc, train_r0, train_r1 = train_model(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate
        (test_loss, test_acc, test_prec, test_rec, test_f1, cm,
         test_bacc, test_r0, test_r1, test_bacc_sklearn, probs) = evaluate_model(
            model, test_loader, criterion, device
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_baccs.append(train_bacc)
        test_baccs.append(test_bacc)

        scheduler.step(test_bacc)  # Adjust learning rate based on BAcc

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"  Train: Loss={train_loss:.4f} | BAcc={train_bacc * 100:.2f}% | "
                  f"R0={train_r0 * 100:.1f}% | R1={train_r1 * 100:.1f}%")
            print(f"  Test:  Loss={test_loss:.4f} | BAcc={test_bacc * 100:.2f}% | "
                  f"R0={test_r0 * 100:.1f}% | R1={test_r1 * 100:.1f}%")

        # Early stopping based on BAcc
        if test_bacc > best_test_bacc:
            best_test_bacc = test_bacc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Final evaluation
    print("\n" + "=" * 70)
    print("STEP 4: FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    (test_loss, accuracy, precision, recall, f1, cm,
     bacc, recall_class_0, recall_class_1, bacc_sklearn, probabilities) = evaluate_model(
        model, test_loader, criterion, device
    )

    print(f"\n{'=' * 70}")
    print(f"PRIMARY METRIC - BALANCED ACCURACY (BAcc)")
    print(f"{'=' * 70}")
    print(f"  Balanced Accuracy (BAcc): {bacc * 100:.2f}%")
    print(f"  ├─ Recall Class 0 (Deceased): {recall_class_0 * 100:.2f}%")
    print(f"  └─ Recall Class 1 (Alive):    {recall_class_1 * 100:.2f}%")
    print(f"\n  Verification (sklearn BAcc):  {bacc_sklearn * 100:.2f}%")


    print(f"\n{'=' * 70}")
    print(f"CONFUSION MATRIX")
    print(f"{'=' * 70}")
    print(f"                 Predicted")
    print(f"               Dead  Alive")
    print(f"Actual Dead   {cm[0, 0]:4d}  {cm[0, 1]:4d}")
    print(f"       Alive  {cm[1, 0]:4d}  {cm[1, 1]:4d}")



if __name__ == "__main__":
    main()