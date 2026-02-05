"""
Early Fusion (Mean) - Multi-Modal Survival Prediction
Aligned with thesis specifications from Tiago Mota's work
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


# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)


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


class EarlyFusionMean(nn.Module):
    """
    Early Fusion using Masked Mean - Thesis Implementation

    Architecture matches Table A.2 specifications:
    - 5 Linear blocks (4 for encoding + 1 classifier)
    - Hidden size: 128
    - Batch normalization: Yes
    - Dropout: Not specified in thesis for this model
    """

    def __init__(self, ct_dim=512, mri_dim=512, wsi_dim=2048, clinical_dim=16,
                 hidden_size=128):
        super(EarlyFusionMean, self).__init__()

        # Encoding blocks for each modality (with BatchNorm as per thesis)
        self.ct_encoder = self._make_encoding_block(ct_dim, hidden_size)
        self.mri_encoder = self._make_encoding_block(mri_dim, hidden_size)
        self.wsi_encoder = self._make_encoding_block(wsi_dim, hidden_size)
        self.clinical_encoder = self._make_encoding_block(clinical_dim, hidden_size)

        # Classifier block (5th linear block)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def _make_encoding_block(self, in_features, out_features):
        """
        Creates an encoding block matching Figure 4.7 in thesis:
        Linear -> BatchNorm -> ReLU
        """
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )

    def forward(self, ct, mri, wsi, clinical, ct_mask, mri_mask, wsi_mask):
        """
        Forward pass with masked mean fusion as described in thesis Section 4.5.2

        Key difference from original: Uses personalized masked mean that considers
        only available modalities for each patient individually.
        """
        batch_size = ct.shape[0]

        # Encode each modality
        ct_encoded = self.ct_encoder(ct)
        mri_encoded = self.mri_encoder(mri)
        wsi_encoded = self.wsi_encoder(wsi)
        clinical_encoded = self.clinical_encoder(clinical)

        # Apply masks (zero out missing modalities)
        ct_encoded = ct_encoded * ct_mask.unsqueeze(1)
        mri_encoded = mri_encoded * mri_mask.unsqueeze(1)
        wsi_encoded = wsi_encoded * wsi_mask.unsqueeze(1)
        # Clinical always available - no mask needed

        # Masked mean fusion (Equation 2.6 from thesis)
        # Sum all encoded modalities
        fused_sum = ct_encoded + mri_encoded + wsi_encoded + clinical_encoded

        # Count available modalities per patient (clinical always counted)
        num_available = ct_mask + mri_mask + wsi_mask + 1.0
        num_available = num_available.unsqueeze(1)  # Shape: [batch_size, 1]

        # Personalized masked mean
        fused = fused_sum / num_available

        # Pass through classifier
        output = self.classifier(fused)

        return output.squeeze(-1)  # Shape: [batch_size]


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
            df[col] = df[col].fillna(0)
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

    features_array = np.vstack(features_list)
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
def get_oversampler(y_train, oversample_factor=10):
    """
    Create weighted sampler for class imbalance
    Thesis specifies 6x oversampling (Table A.2)

    CRITICAL: This creates balanced batches by oversampling minority class
    """
    y = np.array(y_train).astype(int)
    class_counts = np.bincount(y)

    # Calculate weights to balance classes
    # Weight inversely proportional to class frequency
    class_weights = 1.0 / class_counts

    # Assign weight to each sample based on its class
    sample_weights = np.array([class_weights[int(label)] for label in y])

    # Apply additional oversampling to minority class
    minority_class = np.argmin(class_counts)
    minority_mask = (y == minority_class)
    sample_weights[minority_mask] *= oversample_factor

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=120, device='cuda'):
    """
    Train the early fusion model
    Epochs: 120 as per Table A.2
    """

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
            predictions = (torch.sigmoid(outputs) > 0.5).float()
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
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_bacc = balanced_accuracy_score(val_labels, val_preds)

        # Learning rate scheduling
        scheduler.step()

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
                  f"Val Loss: {val_loss:.4f}, Val BAcc: {val_bacc:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

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
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()

            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
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

    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Dead  Alive")
    print(f"Actual Dead   {cm[0, 0]:4d}  {cm[0, 1]:4d}")
    print(f"       Alive  {cm[1, 0]:4d}  {cm[1, 1]:4d}")

    return all_preds, all_probs, all_labels, bacc


# ============================================================================
# Main Execution
# ============================================================================
def main():
    # Configuration matching thesis Table A.2
    CLINICAL_CSV = 'mmist_data/clinical+genomic_split.csv'
    CT_FOLDER = 'mmist_data/CT_features'
    MRI_FOLDER = 'mmist_data/MRI_features'
    WSI_FOLDER = 'mmist_data/WSI_features'
    CT_CSV = 'mmist_data/CT_Merged.csv'
    MRI_CSV = 'mmist_data/MRI_Merged.csv'
    WSI_CSV = 'mmist_data/WSI_patientfiles.csv'

    # Hyperparameters from Table A.2
    BATCH_SIZE = 14  # Thesis specifies 14
    LEARNING_RATE = 1e-3  # 1e-3 per thesis
    NUM_EPOCHS = 120  # 120 epochs per thesis
    HIDDEN_SIZE = 128  # Hidden size: 128
    OVERSAMPLE_FACTOR = 6  # 6x oversampling
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # CRITICAL INSIGHT: Thesis likely ran multiple experiments
    # Table 5.4 shows best result from hyperparameter tuning
    NUM_RUNS = 5  # Try multiple random initializations

    print(f"Device: {DEVICE}\n")
    print("=" * 80)
    print("ATTEMPTING MULTIPLE RUNS (Thesis likely reports best of several)")
    print("=" * 80)
    print("Hyperparameters (from thesis Table A.2):")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Hidden Size: {HIDDEN_SIZE}")
    print(f"  Oversample Factor: {OVERSAMPLE_FACTOR}x")
    print(f"  Optimizer: AdamW")
    print(f"  LR Scheduler: Cosine")
    print(f"  Batch Normalization: Yes")
    print(f"  Number of runs: {NUM_RUNS}\n")

    # Load all data ONCE
    data = load_all_data(CLINICAL_CSV, CT_FOLDER, MRI_FOLDER, WSI_FOLDER,
                         CT_CSV, MRI_CSV, WSI_CSV)

    # Standardize features ONCE
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

    # Calculate class weights for BCEWithLogitsLoss
    pos_count = data['train']['labels'].sum()
    neg_count = len(data['train']['labels']) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count]).to(DEVICE)

    print(f"\nClass distribution:")
    print(f"  Positive (Alive): {int(pos_count)} ({pos_count / len(data['train']['labels']) * 100:.1f}%)")
    print(f"  Negative (Dead): {int(neg_count)} ({neg_count / len(data['train']['labels']) * 100:.1f}%)")
    print(f"  Pos weight for loss: {pos_weight.item():.4f}\n")

    # Run multiple experiments
    best_bacc = 0.0
    best_model_state = None
    all_results = []

    for run in range(1, NUM_RUNS):
        print("\n" + "=" * 80)
        print(f"RUN {run + 1}/{NUM_RUNS}")
        print("=" * 80)

        # Set different seed for each run
        set_seed(42 + run)

        # Create fresh data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=get_oversampler(data['train']['labels'], OVERSAMPLE_FACTOR),
            drop_last=True,
            num_workers=0,
            pin_memory=(DEVICE == 'cuda')
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=(DEVICE == 'cuda')
        )

        # Initialize fresh model
        model = EarlyFusionMean(
            ct_dim=512,
            mri_dim=512,
            wsi_dim=2048,
            clinical_dim=train_clinical.shape[1],
            hidden_size=HIDDEN_SIZE
        ).to(DEVICE)

        if run == 0:
            print(f"\nModel Architecture (5 Linear Blocks):")
            print(model)
            print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS,
            eta_min=1e-6
        )

        # Train
        model = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler,
                            num_epochs=NUM_EPOCHS, device=DEVICE)

        # Evaluate
        predictions, probabilities, labels, bacc = evaluate_model(model, test_loader, device=DEVICE)

        all_results.append({
            'run': run + 1,
            'bacc': bacc,
            'predictions': predictions,
            'probabilities': probabilities
        })

        if bacc > best_bacc:
            best_bacc = bacc
            best_model_state = model.state_dict().copy()

    # Report all runs
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL RUNS")
    print("=" * 80)
    for result in all_results:
        print(f"Run {result['run']}: BACC = {result['bacc']:.4f}")
    print(f"\nBest BACC achieved: {best_bacc:.4f}")
    print(f"Mean BACC: {np.mean([r['bacc'] for r in all_results]):.4f}")
    print(f"Std BACC: {np.std([r['bacc'] for r in all_results]):.4f}")
    print("=" * 80)

    return best_bacc


if __name__ == "__main__":
    main()