"""
Early Fusion (Mean) - Multi-Modal Survival Prediction
======================================================
Combines Clinical+Genomics, CT, MRI, and WSI features using mean pooling.
Handles missing modalities by filling with zeros and masking.
"""

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



# TEST SET RESULTS - EARLY FUSION (MEAN)
# ================================================================================
#
# Balanced Accuracy: 0.7289 (72.89%)


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# Dataset Class
# ============================================================================
class MultiModalDataset(Dataset):
    """Dataset for multi-modal features with availability masks"""

    def __init__(self, ct_features, mri_features, wsi_features, clinical_features, labels,
                 ct_mask, mri_mask, wsi_mask):
        self.ct = torch.FloatTensor(ct_features)
        self.mri = torch.FloatTensor(mri_features)
        self.wsi = torch.FloatTensor(wsi_features)
        self.clinical = torch.FloatTensor(clinical_features)
        self.labels = torch.FloatTensor(labels)

        self.ct_mask = torch.FloatTensor(ct_mask)
        self.mri_mask = torch.FloatTensor(mri_mask)
        self.wsi_mask = torch.FloatTensor(wsi_mask)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'ct': self.ct[idx],
            'mri': self.mri[idx],
            'wsi': self.wsi[idx],
            'clinical': self.clinical[idx],
            'label': self.labels[idx],
            'ct_mask': self.ct_mask[idx],
            'mri_mask': self.mri_mask[idx],
            'wsi_mask': self.wsi_mask[idx]
        }

class EarlyFusionConcat(nn.Module):
    """
    Early Fusion using Concatenation

    Projects each modality to a common dimension, then concatenates them
    (with zero-masking for missing modalities), followed by a deep classifier.
    """

    def __init__(self, ct_dim=512, mri_dim=512, wsi_dim=2048, clinical_dim=16,
                 proj_dim=128, hidden_dims=[512, 256, 128], dropout=0.3,
                 use_layernorm=False):
        super(EarlyFusionConcat, self).__init__()

        # Projection layers for each modality
        self.ct_proj = nn.Linear(ct_dim, proj_dim)
        self.mri_proj = nn.Linear(mri_dim, proj_dim)
        self.wsi_proj = nn.Linear(wsi_dim, proj_dim)
        self.clinical_proj = nn.Linear(clinical_dim, proj_dim)

        self.use_layernorm = use_layernorm

        # Classifier on concatenated features
        classifier_layers = []
        input_dim = proj_dim * 4

        for hidden_dim in hidden_dims:
            classifier_layers.append(nn.Linear(input_dim, hidden_dim))
            if use_layernorm:
                classifier_layers.append(nn.LayerNorm(hidden_dim))
            else:
                classifier_layers.append(nn.BatchNorm1d(hidden_dim))
            classifier_layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden_dim

        classifier_layers.append(nn.Linear(input_dim, 1))
        classifier_layers.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, ct, mri, wsi, clinical, ct_mask, mri_mask, wsi_mask):
        """
        Forward pass with masked concatenation

        Args:
            ct, mri, wsi, clinical: Feature tensors
            ct_mask, mri_mask, wsi_mask: Binary masks (1 if available, 0 if missing)
        """

        # Apply masks to zero-out missing modalities
        ct_proj = self.ct_proj(ct) * ct_mask.unsqueeze(1)
        mri_proj = self.mri_proj(mri) * mri_mask.unsqueeze(1)
        wsi_proj = self.wsi_proj(wsi) * wsi_mask.unsqueeze(1)
        clinical_proj = self.clinical_proj(clinical)  # always available

        # Concatenate along feature dimension
        fused = torch.cat([ct_proj, mri_proj, wsi_proj, clinical_proj], dim=1)

        # Pass through classifier
        output = self.classifier(fused)

        # Ensure output shape is [B] instead of [B, 1]
        return output.view(-1)

class EarlyFusionMean(nn.Module):
    """
    Early Fusion using Masked Mean

    Projects each modality to a common dimension, sums available modalities,
    then divides by the number of available modalities (clinical always counted),
    followed by a classifier.
    """

    def __init__(self, ct_dim=512, mri_dim=512, wsi_dim=2048, clinical_dim=16,
                 proj_dim=128, hidden_dims=[512, 256, 128], dropout=0.3,
                 use_layernorm=False):
        super(EarlyFusionMean, self).__init__()

        # Projection layers
        self.ct_proj = nn.Linear(ct_dim, proj_dim)
        self.mri_proj = nn.Linear(mri_dim, proj_dim)
        self.wsi_proj = nn.Linear(wsi_dim, proj_dim)
        self.clinical_proj = nn.Linear(clinical_dim, proj_dim)

        self.use_layernorm = use_layernorm

        # Classifier
        classifier_layers = []
        input_dim = proj_dim  # after mean fusion, dimension stays proj_dim

        for hidden_dim in hidden_dims:
            classifier_layers.append(nn.Linear(input_dim, hidden_dim))
            if use_layernorm:
                classifier_layers.append(nn.LayerNorm(hidden_dim))
            else:
                classifier_layers.append(nn.BatchNorm1d(hidden_dim))
            classifier_layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden_dim

        classifier_layers.append(nn.Linear(input_dim, 1))
        classifier_layers.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, ct, mri, wsi, clinical, ct_mask, mri_mask, wsi_mask):
        """
        Forward pass with masked mean fusion.

        Args:
            ct, mri, wsi, clinical: Feature tensors
            ct_mask, mri_mask, wsi_mask: Binary masks (1 if available, 0 if missing)
        """

        # Project features
        ct_proj = self.ct_proj(ct) * ct_mask.unsqueeze(1)
        mri_proj = self.mri_proj(mri) * mri_mask.unsqueeze(1)
        wsi_proj = self.wsi_proj(wsi) * wsi_mask.unsqueeze(1)
        clinical_proj = self.clinical_proj(clinical)  # always available

        # Sum projections
        fused_sum = ct_proj + mri_proj + wsi_proj + clinical_proj

        # Count available modalities
        num_available = ct_mask + mri_mask + wsi_mask + 1.0  # +1 for clinical
        num_available = num_available.unsqueeze(1)

        # Masked mean fusion
        fused = fused_sum / num_available

        # Classifier
        output = self.classifier(fused)
        return output.view(-1)


# ============================================================================
# Data Loading
# ============================================================================
def load_clinical_data(clinical_csv):
    """Load clinical and genomics features"""

    df = pd.read_csv(clinical_csv)

    feature_columns = [
        'gender', 'age_diag', 'grade',
        'ajcc_path_tumor_pt', 'ajcc_path_nodes_pn',
        'ajcc_clin_metastasis_cm', 'ajcc_path_metastasis_pm',
        'ajcc_path_tumor_stage',
        'race_Asian', 'race_Black or African American',
        'race_Hispanic or Latino', 'race_White', 'race_other',
        'VHL_mutation', 'PBMR1_mutation', 'TTN_mutation'
    ]

    # Handle missing values
    for col in feature_columns:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)
            df[col] = df[col].fillna(df[col].median())

    return df, feature_columns


def load_imaging_features(case_ids, modality_folder, chosen_exam_csv, modality_name):
    """
    Load imaging features for given case IDs
    Returns features array and availability mask
    """

    chosen_exams = pd.read_csv(chosen_exam_csv)
    chosen_exams_dict = dict(zip(chosen_exams['case_id'], chosen_exams['chosen_exam']))

    features_list = []
    masks = []

    # Define fixed feature dimensions
    if modality_name in ['CT', 'MRI']:
        feature_dim = 512
    elif modality_name == 'WSI':
        feature_dim = 2048
    else:
        raise ValueError(f"Unknown modality: {modality_name}")

    print(f"Loading {modality_name} features...")

    for case_id in case_ids:
        feat_vector = np.zeros(feature_dim)  # default zero vector
        mask = 0.0

        if case_id in chosen_exams_dict:
            npz_file = os.path.join(modality_folder, chosen_exams_dict[case_id])

            if os.path.exists(npz_file):
                try:
                    npz_data = np.load(npz_file)
                    features = npz_data[npz_data.files[0]]

                    # Flatten / squeeze features
                    features = features.squeeze()
                    if len(features.shape) > 1:
                        features = features.mean(axis=0)

                    # Ensure correct dimension
                    if len(features) < feature_dim:
                        features = np.pad(features, (0, feature_dim - len(features)))
                    elif len(features) > feature_dim:
                        features = features[:feature_dim]

                    feat_vector = features
                    mask = 1.0  # available
                except Exception as e:
                    print(f"Error loading {npz_file}: {e}")

        features_list.append(feat_vector)
        masks.append(mask)

    features_array = np.vstack(features_list)  # shape: (num_cases, feature_dim)
    masks_array = np.array(masks)

    available_count = int(masks_array.sum())
    print(
        f"  {modality_name}: {available_count}/{len(case_ids)} available ({available_count / len(case_ids) * 100:.1f}%)"
    )

    return features_array, masks_array



def load_all_data(clinical_csv, ct_folder, mri_folder, wsi_folder,
                  ct_csv, mri_csv, wsi_csv):
    """Load and align all modalities"""

    print("=" * 80)
    print("Loading Multi-Modal Data")
    print("=" * 80)

    # Load clinical data
    clinical_df, feature_columns = load_clinical_data(clinical_csv)

    # Get case IDs for train and test
    train_df = clinical_df[clinical_df['Split'] == 'train'].reset_index(drop=True)
    test_df = clinical_df[clinical_df['Split'] == 'test'].reset_index(drop=True)

    train_case_ids = train_df['case_id'].values
    test_case_ids = test_df['case_id'].values

    print(f"\nDataset: {len(clinical_df)} patients")
    print(f"  Train: {len(train_df)} patients")
    print(f"  Test:  {len(test_df)} patients")

    # Clinical features
    train_clinical = train_df[feature_columns].values.astype(np.float32)
    test_clinical = test_df[feature_columns].values.astype(np.float32)

    # Labels
    train_labels = train_df['vital_status_12'].values.astype(np.float32)
    test_labels = test_df['vital_status_12'].values.astype(np.float32)

    print(f"\nLabel distribution:")
    print(f"  Train: Alive={train_labels.sum()}/{len(train_labels)} ({train_labels.mean() * 100:.1f}%)")
    print(f"  Test:  Alive={test_labels.sum()}/{len(test_labels)} ({test_labels.mean() * 100:.1f}%)")

    # Load imaging features
    print("\nLoading imaging modalities:")
    train_ct, train_ct_mask = load_imaging_features(train_case_ids, ct_folder, ct_csv, 'CT')
    test_ct, test_ct_mask = load_imaging_features(test_case_ids, ct_folder, ct_csv, 'CT')

    train_mri, train_mri_mask = load_imaging_features(train_case_ids, mri_folder, mri_csv, 'MRI')
    test_mri, test_mri_mask = load_imaging_features(test_case_ids, mri_folder, mri_csv, 'MRI')

    train_wsi, train_wsi_mask = load_imaging_features(train_case_ids, wsi_folder, wsi_csv, 'WSI')
    test_wsi, test_wsi_mask = load_imaging_features(test_case_ids, wsi_folder, wsi_csv, 'WSI')

    print(f"\nFeature dimensions:")
    print(f"  Clinical: {train_clinical.shape[1]}")
    print(f"  CT:       {train_ct.shape[1]}")
    print(f"  MRI:      {train_mri.shape[1]}")
    print(f"  WSI:      {train_wsi.shape[1]}")

    return {
        'train': {
            'clinical': train_clinical,
            'ct': train_ct,
            'mri': train_mri,
            'wsi': train_wsi,
            'labels': train_labels,
            'ct_mask': train_ct_mask,
            'mri_mask': train_mri_mask,
            'wsi_mask': train_wsi_mask
        },
        'test': {
            'clinical': test_clinical,
            'ct': test_ct,
            'mri': test_mri,
            'wsi': test_wsi,
            'labels': test_labels,
            'ct_mask': test_ct_mask,
            'mri_mask': test_mri_mask,
            'wsi_mask': test_wsi_mask
        }
    }


# ============================================================================
# Training and Evaluation
# ============================================================================
def get_oversampler(y_train, oversample_factor=6):
    """Create weighted sampler for class imbalance"""
    y = np.array(y_train).astype(int)
    class_counts = np.bincount(y)
    minority = np.argmin(class_counts)

    weights = np.ones(len(y), dtype=float)
    weights[y == minority] *= oversample_factor

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=100, device='cuda'):
    """Train the early fusion model"""

    best_val_bacc = 0.0
    best_model_state = None
    patience = 20
    patience_counter = 0

    print("\n" + "=" * 80)
    print("Training Early Fusion (Mean) Model")
    print("=" * 80)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            ct = batch['ct'].to(device)
            mri = batch['mri'].to(device)
            wsi = batch['wsi'].to(device)
            clinical = batch['clinical'].to(device)
            labels = batch['label'].to(device)

            ct_mask = batch['ct_mask'].to(device)
            mri_mask = batch['mri_mask'].to(device)
            wsi_mask = batch['wsi_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(ct, mri, wsi, clinical, ct_mask, mri_mask, wsi_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_preds.extend(predictions.cpu().detach().numpy())
            train_labels.extend(labels.cpu().detach().numpy())

        train_loss /= len(train_loader)
        train_bacc = balanced_accuracy_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                ct = batch['ct'].to(device)
                mri = batch['mri'].to(device)
                wsi = batch['wsi'].to(device)
                clinical = batch['clinical'].to(device)
                labels = batch['label'].to(device)

                ct_mask = batch['ct_mask'].to(device)
                mri_mask = batch['mri_mask'].to(device)
                wsi_mask = batch['wsi_mask'].to(device)

                outputs = model(ct, mri, wsi, clinical, ct_mask, mri_mask, wsi_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_bacc = balanced_accuracy_score(val_labels, val_preds)

        # Save best model
        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train BAcc: {train_bacc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val BAcc: {val_bacc:.4f}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nBest Validation Balanced Accuracy: {best_val_bacc:.4f}")
    return model


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set"""

    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            ct = batch['ct'].to(device)
            mri = batch['mri'].to(device)
            wsi = batch['wsi'].to(device)
            clinical = batch['clinical'].to(device)
            labels = batch['label']

            ct_mask = batch['ct_mask'].to(device)
            mri_mask = batch['mri_mask'].to(device)
            wsi_mask = batch['wsi_mask'].to(device)

            outputs = model(ct, mri, wsi, clinical, ct_mask, mri_mask, wsi_mask)
            predictions = (outputs > 0.5).float()

            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Calculate metrics
    bacc = balanced_accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 80)
    print("TEST SET RESULTS - EARLY FUSION (MEAN)")
    print("=" * 80)
    print(f"\nBalanced Accuracy: {bacc:.4f} ({bacc * 100:.2f}%)")

    # print("\nClassification Report:")
    # print(classification_report(all_labels, all_preds,
    #                             target_names=['Deceased', 'Alive']))

    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Dead  Alive")
    print(f"Actual Dead   {cm[0, 0]:4d}  {cm[0, 1]:4d}")
    print(f"       Alive  {cm[1, 0]:4d}  {cm[1, 1]:4d}")

    # try:
    #     auc = roc_auc_score(all_labels, all_probs)
    #     print(f"\nROC AUC Score: {auc:.4f}")
    # except:
    #     print("\nROC AUC Score: Could not be calculated")

    return all_preds, all_probs, all_labels, bacc


# ============================================================================
# Main Execution
# ============================================================================
def main():
    # Configuration
    CLINICAL_CSV = 'mmist_data/clinical+genomic_split.csv'
    CT_FOLDER = 'mmist_data/CT_features'
    MRI_FOLDER = 'mmist_data/MRI_features'
    WSI_FOLDER = 'mmist_data/WSI_features'
    CT_CSV = 'mmist_data/CT_Merged.csv'
    MRI_CSV = 'mmist_data/MRI_Merged.csv'
    WSI_CSV = 'mmist_data/WSI_patientfiles.csv'

    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 120
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {DEVICE}\n")

    # Load all data
    data = load_all_data(CLINICAL_CSV, CT_FOLDER, MRI_FOLDER, WSI_FOLDER,
                         CT_CSV, MRI_CSV, WSI_CSV)

    # Standardize features
    print("\nStandardizing features...")
    clinical_scaler = StandardScaler()
    ct_scaler = StandardScaler()
    mri_scaler = StandardScaler()
    wsi_scaler = StandardScaler()

    train_clinical = clinical_scaler.fit_transform(data['train']['clinical'])
    test_clinical = clinical_scaler.transform(data['test']['clinical'])

    train_ct = ct_scaler.fit_transform(data['train']['ct'])
    test_ct = ct_scaler.transform(data['test']['ct'])

    train_mri = mri_scaler.fit_transform(data['train']['mri'])
    test_mri = mri_scaler.transform(data['test']['mri'])

    train_wsi = wsi_scaler.fit_transform(data['train']['wsi'])
    test_wsi = wsi_scaler.transform(data['test']['wsi'])

    # Create datasets
    train_dataset = MultiModalDataset(
        train_ct, train_mri, train_wsi, train_clinical,
        data['train']['labels'],
        data['train']['ct_mask'],
        data['train']['mri_mask'],
        data['train']['wsi_mask']
    )

    test_dataset = MultiModalDataset(
        test_ct, test_mri, test_wsi, test_clinical,
        data['test']['labels'],
        data['test']['ct_mask'],
        data['test']['mri_mask'],
        data['test']['wsi_mask']
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=get_oversampler(data['train']['labels']),
        drop_last= True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Initialize model
    model = EarlyFusionMean(
        ct_dim=512,
        mri_dim=512,
        wsi_dim=2048,
        clinical_dim=train_clinical.shape[1],
        proj_dim=256,
        hidden_dims=[128],
        dropout=0.3
    ).to(DEVICE)

    # print(f"\nModel Architecture:")
    # print(model)
    # print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Train
    model = train_model(model, train_loader, test_loader, criterion, optimizer,
                        num_epochs=NUM_EPOCHS, device=DEVICE)

    # Evaluate
    predictions, probabilities, labels, bacc = evaluate_model(model, test_loader, device=DEVICE)

    # # Save model
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'clinical_scaler': clinical_scaler,
    #     'ct_scaler': ct_scaler,
    #     'mri_scaler': mri_scaler,
    #     'wsi_scaler': wsi_scaler,
    #     'test_bacc': bacc
    # }, 'early_fusion_mean_model.pth')
    #
    # print("\n" + "=" * 80)
    # print("Model saved to 'early_fusion_mean_model.pth'")
    # print("=" * 80)

    return bacc


if __name__ == "__main__":
    main()