import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score
import os
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# === Test Set Results ===
#
# Balanced Accuracy: 0.5509
#
# Confusion Matrix:
# [[ 1  3]
#  [ 8 46]]

class CTDataset(Dataset):
    """Custom Dataset for loading CT features"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SurvivalCT(nn.Module):
    """Multi-layer Perceptron for survival prediction using 512-dimensional CT features"""

    def __init__(self, feature_dim=512, hidden_dims=[256, 128, 64], dropout=0.3):
        super(SurvivalCT, self).__init__()

        layers = []
        prev_dim = feature_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # output probability

        self.network = nn.Sequential(*layers)

    def forward(self, features):
        output = self.network(features)
        return output.squeeze()


def load_data(ct_folder, chosen_exam_csv, clinical_csv):
    """Load and prepare data from files"""

    # Load CSV files
    chosen_exams = pd.read_csv(chosen_exam_csv)
    clinical_data = pd.read_csv(clinical_csv)

    # Merge data to get labels and split info
    data = pd.merge(chosen_exams, clinical_data[['case_id', 'vital_status_12', 'Split']], on='case_id')

    # Load CT features
    ct_features_list = []
    case_ids = []

    print("Loading CT features...")
    for idx, row in data.iterrows():

        npz_file = os.path.join(ct_folder, row['chosen_exam'])

        if os.path.exists(npz_file):
            npz_data = np.load(npz_file)
            # Assuming the features are stored in the first array
            features = npz_data[npz_data.files[0]]

            # If features are 2D, aggregate them (e.g., mean pooling)
            if len(features.shape) > 1:
                features = features.mean(axis=0)

            ct_features_list.append(features)
            case_ids.append(row['case_id'])
        else:
            print(f"Warning: File not found - {npz_file}")

    ct_features_array = np.array(ct_features_list)
    # np.save('CT_features.npy', ct_features_array)
    # Filter data to only include cases with loaded features
    data = data[data['case_id'].isin(case_ids)].reset_index(drop=True)

    labels = data['vital_status_12'].values
    ct_features = np.array(ct_features_list)

    # Split into train and test based on 'Split' column
    train_mask = data['Split'] == 'train'
    test_mask = data['Split'] == 'test'

    return {
        'train': {
            'ct': ct_features[train_mask],
            'labels': labels[train_mask]
        },
        'test': {
            'ct': ct_features[test_mask],
            'labels': labels[test_mask]
        }
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    """Train the MLP model"""

    best_val_balanced_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_preds.extend(predictions.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_balanced_acc = balanced_accuracy_score(train_labels, train_preds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_balanced_acc = balanced_accuracy_score(val_labels, val_preds)

        # Save best model based on balanced accuracy
        if val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = val_balanced_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Bal Acc: {train_balanced_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Bal Acc: {val_balanced_acc:.4f}')

    # Load best model
    model.load_state_dict(best_model_state)
    print(f'\nBest Validation Balanced Accuracy: {best_val_balanced_acc:.4f}')
    return model


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate the model on test set"""

    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)

            outputs = model(features)
            predictions = (outputs > 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)

    # Calculate metrics
    balanced_acc = balanced_accuracy_score (all_labels, all_predictions)

    print("\n=== Test Set Results ===")
    print(f"\nBalanced Accuracy: {balanced_acc:.4f}")

    # print("\nClassification Report:")
    # print(classification_report(all_labels, all_predictions, target_names=['Survived', 'Deceased']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))

    # try:
    #     auc_score = roc_auc_score(all_labels, all_probabilities)
    #     print(f"\nROC AUC Score: {auc_score:.4f}")
    # except:
    #     print("\nROC AUC Score: Could not be calculated")

    return all_predictions, all_probabilities, all_labels

def get_oversampler(y_train, oversample_factor=6):
    y = np.array(y_train).astype(int)
    class_counts = np.bincount(y)
    minority = np.argmin(class_counts)

    # Base weight = 1, minority weight = oversample_factor
    weights = np.ones(len(y), dtype=float)
    weights[y == minority] *= oversample_factor

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def main():
    # Configuration
    CT_FOLDER = 'mmist_data/CT_features'
    CHOSEN_EXAM_CSV = 'mmist_data/CT_Merged.csv'
    CLINICAL_CSV = 'mmist_data/clinical+genomic_split.csv' #for labels

    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")

    # Load data
    print("\nLoading data...")
    data = load_data(CT_FOLDER, CHOSEN_EXAM_CSV, CLINICAL_CSV)


    print(f"Train samples: {len(data['train']['labels'])}")
    print(f"Test samples: {len(data['test']['labels'])}")
    print(f"CT feature dimension: {data['train']['ct'].shape[2]}")

    # Standardize features

    # Remove singleton dimensions
    train_ct_pooled = np.squeeze(data['train']['ct'], axis=(1, 3))
    test_ct_pooled = np.squeeze(data['test']['ct'], axis=(1, 3))

    ct_scaler = StandardScaler()
    train_ct = ct_scaler.fit_transform(train_ct_pooled)
    test_ct = ct_scaler.transform(test_ct_pooled)

    # Create datasets and dataloaders
    train_dataset = CTDataset(train_ct, data['train']['labels'])
    test_dataset = CTDataset(test_ct, data['test']['labels'])

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler= get_oversampler(data['train']['labels'], oversample_factor=6,),
                              drop_last = True) # ← oversampling happens here )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = SurvivalCT(
        feature_dim=train_ct.shape[1],
        hidden_dims=[512, 256, 128],
        dropout=0.3
    ).to(DEVICE)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Train model
    print("\nTraining model...")
    model = train_model(model, train_loader, test_loader, criterion, optimizer,
                        num_epochs=NUM_EPOCHS, device=DEVICE)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions, probabilities, labels = evaluate_model(model, test_loader, device=DEVICE)




if __name__ == "__main__":
    main()