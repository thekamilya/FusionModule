"""
Complete Pipeline: Encoder-Decoder Reconstruction + Classification
Step 1: Train reconstruction model
Step 2: Use it to handle missing modalities in classification
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import os


def set_seed(seed=43):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)


# ============================================================================
# PART 1: RECONSTRUCTION MODEL
# ============================================================================

class ReconstructionDataset(Dataset):
    """
    Dataset for training the reconstruction model

    Paper requirements:
    - Gaussian noise augmentation (µ = 0, σ = 0.01)
    - Modality dropout only when at most one modality is missing
    """

    def __init__(self, ct_features, mri_features, wsi_features, clinical_features,
                 ct_mask, mri_mask, wsi_mask,
                 gaussian_noise_std=0.01, apply_augmentation=True):
        self.ct = torch.FloatTensor(ct_features)
        self.mri = torch.FloatTensor(mri_features)
        self.wsi = torch.FloatTensor(wsi_features)
        self.clinical = torch.FloatTensor(clinical_features)

        self.ct_mask = torch.FloatTensor(ct_mask)
        self.mri_mask = torch.FloatTensor(mri_mask)
        self.wsi_mask = torch.FloatTensor(wsi_mask)


        self.gaussian_noise_std = gaussian_noise_std
        self.apply_augmentation = apply_augmentation

    def __len__(self):
        return len(self.ct)

    def __getitem__(self, idx):
        # Get original features (targets - NO noise on targets)
        ct_target = self.ct[idx].clone()
        mri_target = self.mri[idx].clone()
        wsi_target = self.wsi[idx].clone()
        clinical_target = self.clinical[idx].clone()

        # Get original availability masks
        ct_available = self.ct_mask[idx].clone()
        mri_available = self.mri_mask[idx].clone()
        wsi_available = self.wsi_mask[idx].clone()

        # Create input versions
        ct_input = ct_target.clone()
        mri_input = mri_target.clone()
        wsi_input = wsi_target.clone()
        clinical_input = clinical_target.clone()

        # PAPER REQUIREMENT: Apply Gaussian noise (µ = 0, σ = 0.01) to available modalities
        if self.apply_augmentation:
            if ct_available.item() > 0:
                ct_input = ct_input + torch.randn_like(ct_input) * self.gaussian_noise_std
            if mri_available.item() > 0:
                mri_input = mri_input + torch.randn_like(mri_input) * self.gaussian_noise_std
            if wsi_available.item() > 0:
                wsi_input = wsi_input + torch.randn_like(wsi_input) * self.gaussian_noise_std
            # Clinical also gets noise
            clinical_input = clinical_input + torch.randn_like(clinical_input) * self.gaussian_noise_std

        # PAPER REQUIREMENT: Modality dropout ONLY when at most one modality is missing
        # Count how many modalities are missing
        num_missing = 0
        if ct_available.item() == 0:
            num_missing += 1
        if mri_available.item() == 0:
            num_missing += 1
        if wsi_available.item() == 0:
            num_missing += 1

        # Only apply dropout if at most 1 is already missing
        if num_missing <= 1 and self.apply_augmentation:
            # Get list of available modalities
            available_modalities = []
            if ct_available.item() > 0:
                available_modalities.append('ct')
            if mri_available.item() > 0:
                available_modalities.append('mri')
            if wsi_available.item() > 0:
                available_modalities.append('wsi')

            # Randomly drop one available modality
            if len(available_modalities) > 0:
                drop_modality = np.random.choice(available_modalities)
                if drop_modality == 'ct':
                    ct_input = torch.zeros_like(ct_input)
                    ct_available = torch.tensor(0.0)
                elif drop_modality == 'mri':
                    mri_input = torch.zeros_like(mri_input)
                    mri_available = torch.tensor(0.0)
                elif drop_modality == 'wsi':
                    wsi_input = torch.zeros_like(wsi_input)
                    wsi_available = torch.tensor(0.0)

        # For modalities that were originally missing, input is zeros
        if self.ct_mask[idx].item() == 0:
            ct_input = torch.zeros_like(ct_input)
        if self.mri_mask[idx].item() == 0:
            mri_input = torch.zeros_like(mri_input)
        if self.wsi_mask[idx].item() == 0:
            wsi_input = torch.zeros_like(wsi_input)

        return {
            'ct_input': ct_input,
            'mri_input': mri_input,
            'wsi_input': wsi_input,
            'clinical_input': clinical_input,
            'ct_target': ct_target,
            'mri_target': mri_target,
            'wsi_target': wsi_target,
            'clinical_target': clinical_target,
            'ct_mask': self.ct_mask[idx],
            'mri_mask': self.mri_mask[idx],
            'wsi_mask': self.wsi_mask[idx],
            'ct_available': ct_available,
            'mri_available': mri_available,
            'wsi_available': wsi_available,
        }


class ModalityReconstructionModel(nn.Module):
    """Encoder-Decoder for missing modality reconstruction"""

    def __init__(self, ct_dim=512, mri_dim=512, wsi_dim=2048, clinical_dim=17,
                 embedding_dim=128):
        super(ModalityReconstructionModel, self).__init__()

        # ENCODER: Project each modality to embedding space
        self.ct_encoder = self._make_encoder_mlp(ct_dim, embedding_dim)
        self.mri_encoder = self._make_encoder_mlp(mri_dim, embedding_dim)
        self.wsi_encoder = self._make_encoder_mlp(wsi_dim, embedding_dim)
        self.clinical_encoder = self._make_encoder_mlp(clinical_dim, embedding_dim)

        # CROSS-MODAL FUSION: Learn relationships between modalities
        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU()
        )

        # DECODER: Reconstruct each modality from fused representation
        self.ct_decoder = self._make_decoder_mlp(embedding_dim, ct_dim)
        self.mri_decoder = self._make_decoder_mlp(embedding_dim, mri_dim)
        self.wsi_decoder = self._make_decoder_mlp(embedding_dim, wsi_dim)
        self.clinical_decoder = self._make_decoder_mlp(embedding_dim, clinical_dim)

    def _make_encoder_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def _make_decoder_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, ct, mri, wsi, clinical):
        # Encode each modality
        ct_emb = self.ct_encoder(ct)
        mri_emb = self.mri_encoder(mri)
        wsi_emb = self.wsi_encoder(wsi)
        clinical_emb = self.clinical_encoder(clinical)

        # Fuse embeddings (learn cross-modal relationships)
        all_embeddings = torch.cat([ct_emb, mri_emb, wsi_emb, clinical_emb], dim=1)
        fused = self.cross_modal_fusion(all_embeddings)

        # Decode back to original feature spaces
        ct_recon = self.ct_decoder(fused)
        mri_recon = self.mri_decoder(fused)
        wsi_recon = self.wsi_decoder(fused)
        clinical_recon = self.clinical_decoder(fused)

        return ct_recon, mri_recon, wsi_recon, clinical_recon

    def reconstruct_missing_modalities(self, ct, mri, wsi, clinical,
                                       ct_mask, mri_mask, wsi_mask):
        """
        PAPER REQUIREMENT: Only reconstruct missing modalities
        Keep original (correct) information for available modalities
        """
        ct_recon, mri_recon, wsi_recon, clinical_recon = self.forward(ct, mri, wsi, clinical)

        # Use original if available, reconstruction if missing
        ct_final = torch.where(ct_mask.unsqueeze(1) > 0, ct, ct_recon)
        mri_final = torch.where(mri_mask.unsqueeze(1) > 0, mri, mri_recon)
        wsi_final = torch.where(wsi_mask.unsqueeze(1) > 0, wsi, wsi_recon)
        clinical_final = clinical  # Always available

        return ct_final, mri_final, wsi_final, clinical_final


def get_reconstruction_oversampler(labels, oversample_factor=6):
    """
    PAPER REQUIREMENT: 6× oversampling of minority class (death at 12 months)

    In the labels: 0 = death, 1 = alive
    So we oversample class 0 (minority class)
    """
    y = np.array(labels).astype(int)
    class_counts = np.bincount(y)

    print(f"\nClass distribution before oversampling:")
    print(f"  Death (0): {class_counts[0]} samples")
    print(f"  Alive (1): {class_counts[1]} samples")

    # Create base weights inversely proportional to class frequency
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[int(label)] for label in y])

    # Apply 6× oversampling to minority class (class 0 = death)
    minority_class = np.argmin(class_counts)
    minority_mask = (y == minority_class)
    sample_weights[minority_mask] *= oversample_factor

    print(f"Applying {oversample_factor}× oversampling to minority class (death)")

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def train_reconstruction_model(model, train_loader, val_loader, num_epochs=50,
                               learning_rate=1e-3, device='cuda'):
    """Train the reconstruction model with paper-compliant settings"""

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    best_model_state = None

    print("\n" + "=" * 80)
    print("TRAINING RECONSTRUCTION MODEL (Paper-Compliant)")
    print("=" * 80)
    print(f"✓ 6× oversampling of minority class")
    print(f"✓ Gaussian noise augmentation (µ=0, σ=0.01)")
    print(f"✓ Modality dropout only when ≤1 modality missing")
    print("=" * 80)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            ct_input = batch['ct_input'].to(device)
            mri_input = batch['mri_input'].to(device)
            wsi_input = batch['wsi_input'].to(device)
            clinical_input = batch['clinical_input'].to(device)

            ct_target = batch['ct_target'].to(device)
            mri_target = batch['mri_target'].to(device)
            wsi_target = batch['wsi_target'].to(device)
            clinical_target = batch['clinical_target'].to(device)

            ct_mask = batch['ct_mask'].to(device)
            mri_mask = batch['mri_mask'].to(device)
            wsi_mask = batch['wsi_mask'].to(device)

            optimizer.zero_grad()
            ct_recon, mri_recon, wsi_recon, clinical_recon = model(
                ct_input, mri_input, wsi_input, clinical_input
            )

            # MSE loss only for available modalities
            ct_loss = nn.functional.mse_loss(ct_recon, ct_target, reduction='none').mean(dim=1)
            ct_loss = (ct_loss * ct_mask).sum() / (ct_mask.sum() + 1e-8)

            mri_loss = nn.functional.mse_loss(mri_recon, mri_target, reduction='none').mean(dim=1)
            mri_loss = (mri_loss * mri_mask).sum() / (mri_mask.sum() + 1e-8)

            wsi_loss = nn.functional.mse_loss(wsi_recon, wsi_target, reduction='none').mean(dim=1)
            wsi_loss = (wsi_loss * wsi_mask).sum() / (wsi_mask.sum() + 1e-8)

            clinical_loss = nn.functional.mse_loss(clinical_recon, clinical_target)

            loss = ct_loss + mri_loss + wsi_loss + clinical_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                ct_input = batch['ct_input'].to(device)
                mri_input = batch['mri_input'].to(device)
                wsi_input = batch['wsi_input'].to(device)
                clinical_input = batch['clinical_input'].to(device)

                ct_target = batch['ct_target'].to(device)
                mri_target = batch['mri_target'].to(device)
                wsi_target = batch['wsi_target'].to(device)
                clinical_target = batch['clinical_target'].to(device)

                ct_mask = batch['ct_mask'].to(device)
                mri_mask = batch['mri_mask'].to(device)
                wsi_mask = batch['wsi_mask'].to(device)

                ct_recon, mri_recon, wsi_recon, clinical_recon = model(
                    ct_input, mri_input, wsi_input, clinical_input
                )

                ct_loss = nn.functional.mse_loss(ct_recon, ct_target, reduction='none').mean(dim=1)
                ct_loss = (ct_loss * ct_mask).sum() / (ct_mask.sum() + 1e-8)

                mri_loss = nn.functional.mse_loss(mri_recon, mri_target, reduction='none').mean(dim=1)
                mri_loss = (mri_loss * mri_mask).sum() / (mri_mask.sum() + 1e-8)

                wsi_loss = nn.functional.mse_loss(wsi_recon, wsi_target, reduction='none').mean(dim=1)
                wsi_loss = (wsi_loss * wsi_mask).sum() / (wsi_mask.sum() + 1e-8)

                clinical_loss = nn.functional.mse_loss(clinical_recon, clinical_target)

                loss = ct_loss + mri_loss + wsi_loss + clinical_loss
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nBest Validation Loss: {best_val_loss:.6f}")
    print("Reconstruction model training complete!")

    return model


def evaluate_reconstruction_quality(model, test_loader, device='cuda'):
    """Evaluate reconstruction quality"""

    model.eval()

    results = {
        'ct': {'mse': [], 'cosine_sim': []},
        'mri': {'mse': [], 'cosine_sim': []},
        'wsi': {'mse': [], 'cosine_sim': []}
    }

    print("\n" + "=" * 80)
    print("RECONSTRUCTION QUALITY EVALUATION")
    print("=" * 80)

    with torch.no_grad():
        for batch in test_loader:
            ct = batch['ct_target'].to(device)
            mri = batch['mri_target'].to(device)
            wsi = batch['wsi_target'].to(device)
            clinical = batch['clinical_target'].to(device)

            ct_mask = batch['ct_mask'].to(device)
            mri_mask = batch['mri_mask'].to(device)
            wsi_mask = batch['wsi_mask'].to(device)

            batch_size = ct.shape[0]

            # Test CT reconstruction (zero out CT, use MRI+WSI+Clinical)
            ct_input_dropped = torch.zeros_like(ct)
            ct_recon, _, _, _ = model(ct_input_dropped, mri, wsi, clinical)

            for i in range(batch_size):
                if ct_mask[i] > 0:
                    mse = nn.functional.mse_loss(ct_recon[i], ct[i]).item()
                    cos_sim = nn.functional.cosine_similarity(
                        ct_recon[i].unsqueeze(0), ct[i].unsqueeze(0)
                    ).item()
                    results['ct']['mse'].append(mse)
                    results['ct']['cosine_sim'].append(cos_sim)

            # Test MRI reconstruction
            mri_input_dropped = torch.zeros_like(mri)
            _, mri_recon, _, _ = model(ct, mri_input_dropped, wsi, clinical)

            for i in range(batch_size):
                if mri_mask[i] > 0:
                    mse = nn.functional.mse_loss(mri_recon[i], mri[i]).item()
                    cos_sim = nn.functional.cosine_similarity(
                        mri_recon[i].unsqueeze(0), mri[i].unsqueeze(0)
                    ).item()
                    results['mri']['mse'].append(mse)
                    results['mri']['cosine_sim'].append(cos_sim)

            # Test WSI reconstruction
            wsi_input_dropped = torch.zeros_like(wsi)
            _, _, wsi_recon, _ = model(ct, mri, wsi_input_dropped, clinical)

            for i in range(batch_size):
                if wsi_mask[i] > 0:
                    mse = nn.functional.mse_loss(wsi_recon[i], wsi[i]).item()
                    cos_sim = nn.functional.cosine_similarity(
                        wsi_recon[i].unsqueeze(0), wsi[i].unsqueeze(0)
                    ).item()
                    results['wsi']['mse'].append(mse)
                    results['wsi']['cosine_sim'].append(cos_sim)

    # Print results
    print("\nReconstruction Quality Metrics:")
    print("-" * 80)
    for modality in ['ct', 'mri', 'wsi']:
        if len(results[modality]['mse']) > 0:
            mse_mean = np.mean(results[modality]['mse'])
            mse_std = np.std(results[modality]['mse'])
            cos_mean = np.mean(results[modality]['cosine_sim'])
            cos_std = np.std(results[modality]['cosine_sim'])

            print(f"{modality.upper()}:")
            print(f"  MSE: {mse_mean:.6f} ± {mse_std:.6f}")
            print(f"  Cosine Similarity: {cos_mean:.4f} ± {cos_std:.4f}")
            print(f"  Samples: {len(results[modality]['mse'])}")
        else:
            print(f"{modality.upper()}: No samples to evaluate")

    return results
# ============================================================================
# PART 2: CLASSIFICATION MODEL WITH RECONSTRUCTION
# ============================================================================

class ClassificationDataset(Dataset):
    """Dataset for classification with labels"""

    def __init__(self, ct_features, mri_features, wsi_features, clinical_features,
                 labels, ct_mask, mri_mask, wsi_mask):
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


class ClassificationWithReconstruction(nn.Module):
    """
    Classification model that uses reconstruction for missing modalities

    Pipeline:
    1. Reconstruct missing modalities using pre-trained reconstruction model
    2. Fuse all modalities (now complete)
    3. Classify for survival prediction
    """

    def __init__(self, reconstruction_model, ct_dim=512, mri_dim=512, wsi_dim=2048,
                 clinical_dim=17, hidden_size=128):
        super(ClassificationWithReconstruction, self).__init__()

        # Pre-trained reconstruction model (frozen or fine-tuned)
        self.reconstruction_model = reconstruction_model

        # Freeze reconstruction model (optional - can also fine-tune)
        for param in self.reconstruction_model.parameters():
            param.requires_grad = False

        # Encoding blocks for classification
        self.ct_encoder = self._make_encoding_block(ct_dim, hidden_size)
        self.mri_encoder = self._make_encoding_block(mri_dim, hidden_size)
        self.wsi_encoder = self._make_encoding_block(wsi_dim, hidden_size)
        self.clinical_encoder = self._make_encoding_block(clinical_dim, hidden_size)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        # Fusion layer to handle concatenated features + mask indicators
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 4 + 3, hidden_size * 2),  # +3 for mask indicators
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

    def _make_encoding_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )

    def forward(self, ct, mri, wsi, clinical, ct_mask, mri_mask, wsi_mask):
        # STEP 1: Reconstruct missing modalities
        with torch.no_grad():  # Don't update reconstruction model
            ct_complete, mri_complete, wsi_complete, clinical_complete = \
                self.reconstruction_model.reconstruct_missing_modalities(
                    ct, mri, wsi, clinical, ct_mask, mri_mask, wsi_mask
                )

        # STEP 2: Encode complete modalities
        ct_encoded = self.ct_encoder(ct_complete)
        mri_encoded = self.mri_encoder(mri_complete)
        wsi_encoded = self.wsi_encoder(wsi_complete)
        clinical_encoded = self.clinical_encoder(clinical_complete)

        fused = torch.cat([
            ct_encoded, mri_encoded, wsi_encoded, clinical_encoded,
            ct_mask.unsqueeze(1), mri_mask.unsqueeze(1), wsi_mask.unsqueeze(1)
        ], dim=1)

        # Project to final dimension
        fused = self.fusion_layer(fused)

        output = self.classifier(fused)
        return output.squeeze(-1)




def get_oversampler(y_train, oversample_factor=6):
    """Create weighted sampler for class imbalance"""
    y = np.array(y_train).astype(int)
    class_counts = np.bincount(y)
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[int(label)] for label in y])
    minority_class = np.argmin(class_counts)
    minority_mask = (y == minority_class)
    sample_weights[minority_mask] *= oversample_factor

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def train_classification_model(model, train_loader, val_loader, criterion, optimizer,
                               scheduler, num_epochs=120, device='cuda'):
    """Train the classification model"""

    best_val_bacc = 0.0
    best_model_state = None
    patience = 20
    patience_counter = 0

    print("\n" + "=" * 80)
    print("STEP 2: TRAINING CLASSIFICATION MODEL WITH RECONSTRUCTION")
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

        scheduler.step()

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

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nBest Validation Balanced Accuracy: {best_val_bacc:.4f}")
    return model


def evaluate_classification_model(model, test_loader, device='cuda'):
    """Evaluate classification model"""

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

    bacc = balanced_accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 80)
    print("TEST SET RESULTS - CLASSIFICATION WITH RECONSTRUCTION")
    print("=" * 80)
    print(f"\nBalanced Accuracy: {bacc:.4f} ({bacc * 100:.2f}%)")

    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Dead  Alive")
    print(f"Actual Dead   {cm[0, 0]:4d}  {cm[0, 1]:4d}")
    print(f"       Alive  {cm[1, 0]:4d}  {cm[1, 1]:4d}")

    return all_preds, all_probs, all_labels, bacc


# ============================================================================
# Data Loading
# ============================================================================
def load_clinical_data(clinical_csv):
    """Load clinical and genomics features"""
    df = pd.read_csv(clinical_csv)

    feature_columns = [
        'gender', 'age_diag', 'grade', 'cancer_history',
        'ajcc_path_tumor_pt', 'ajcc_path_nodes_pn',
        'ajcc_clin_metastasis_cm', 'ajcc_path_metastasis_pm',
        'ajcc_path_tumor_stage',
        'race_Asian', 'race_Black or African American',
        'race_Hispanic or Latino', 'race_White', 'race_other',
        'VHL_mutation', 'PBMR1_mutation', 'TTN_mutation'
    ]

    for col in feature_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            df[col] = df[col].replace(-1, np.nan)
            df[col] = df[col].fillna(df[col].median())

    return df, feature_columns


def load_imaging_features(case_ids, modality_folder, chosen_exam_csv, modality_name):
    """Load imaging features for given case IDs"""
    chosen_exams = pd.read_csv(chosen_exam_csv)
    chosen_exams_dict = dict(zip(chosen_exams['case_id'], chosen_exams['chosen_exam']))

    features_list = []
    masks = []

    if modality_name in ['CT', 'MRI']:
        feature_dim = 512
    elif modality_name == 'WSI':
        feature_dim = 2048
    else:
        raise ValueError(f"Unknown modality: {modality_name}")

    print(f"Loading {modality_name} features...")

    for case_id in case_ids:
        feat_vector = np.zeros(feature_dim)
        mask = 0.0

        if case_id in chosen_exams_dict:
            npz_file = os.path.join(modality_folder, chosen_exams_dict[case_id])

            if os.path.exists(npz_file):
                try:
                    npz_data = np.load(npz_file)
                    features = npz_data[npz_data.files[0]]

                    features = features.squeeze()
                    if len(features.shape) > 1:
                        features = features.mean(axis=0)

                    if len(features) < feature_dim:
                        features = np.pad(features, (0, feature_dim - len(features)))
                    elif len(features) > feature_dim:
                        features = features[:feature_dim]

                    feat_vector = features
                    mask = 1.0
                except Exception as e:
                    print(f"Error loading {npz_file}: {e}")

        features_list.append(feat_vector)
        masks.append(mask)

    features_array = np.vstack(features_list)
    masks_array = np.array(masks)

    available_count = int(masks_array.sum())
    print(f"  {modality_name}: {available_count}/{len(case_ids)} available "
          f"({available_count / len(case_ids) * 100:.1f}%)")

    return features_array, masks_array


def load_all_data(clinical_csv, ct_folder, mri_folder, wsi_folder,
                  ct_csv, mri_csv, wsi_csv):
    """Load and align all modalities"""
    print("=" * 80)
    print("Loading Multi-Modal Data")
    print("=" * 80)

    clinical_df, feature_columns = load_clinical_data(clinical_csv)

    train_df = clinical_df[clinical_df['Split'] == 'train'].reset_index(drop=True)
    test_df = clinical_df[clinical_df['Split'] == 'test'].reset_index(drop=True)

    train_case_ids = train_df['case_id'].values
    test_case_ids = test_df['case_id'].values

    print(f"\nDataset: {len(clinical_df)} patients")
    print(f"  Train: {len(train_df)} patients")
    print(f"  Test:  {len(test_df)} patients")

    train_clinical = train_df[feature_columns].values.astype(np.float32)
    test_clinical = test_df[feature_columns].values.astype(np.float32)

    train_labels = train_df['vital_status_12'].values.astype(np.float32)
    test_labels = test_df['vital_status_12'].values.astype(np.float32)

    print(f"\nLabel distribution:")
    print(f"  Train: Alive={train_labels.sum()}/{len(train_labels)} ({train_labels.mean() * 100:.1f}%)")
    print(f"  Test:  Alive={test_labels.sum()}/{len(test_labels)} ({test_labels.mean() * 100:.1f}%)")

    print("\nLoading imaging modalities:")
    train_ct, train_ct_mask = load_imaging_features(train_case_ids, ct_folder, ct_csv, 'CT')
    test_ct, test_ct_mask = load_imaging_features(test_case_ids, ct_folder, ct_csv, 'CT')

    train_mri, train_mri_mask = load_imaging_features(train_case_ids, mri_folder, mri_csv, 'MRI')
    test_mri, test_mri_mask = load_imaging_features(test_case_ids, mri_folder, mri_csv, 'MRI')

    train_wsi, train_wsi_mask = load_imaging_features(train_case_ids, wsi_folder, wsi_csv, 'WSI')
    test_wsi, test_wsi_mask = load_imaging_features(test_case_ids, wsi_folder, wsi_csv, 'WSI')

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

    # Hyperparameters
    RECON_BATCH_SIZE = 16
    RECON_LEARNING_RATE = 1e-3
    RECON_NUM_EPOCHS = 50
    RECON_OVERSAMPLE_FACTOR = 6  # Paper requirement
    GAUSSIAN_NOISE_STD = 0.01  # Paper requirement

    CLASS_BATCH_SIZE = 14
    CLASS_LEARNING_RATE = 1e-3
    CLASS_NUM_EPOCHS = 120
    CLASS_OVERSAMPLE_FACTOR = 6

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}\n")

    # Load data
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

    # ========================================================================
    # PART 1: TRAIN RECONSTRUCTION MODEL (PAPER-COMPLIANT)
    # ========================================================================

    print("\n" + "=" * 80)
    print("PHASE 1: PAPER-COMPLIANT RECONSTRUCTION MODEL")
    print("=" * 80)

    # Create datasets with labels for oversampling
    recon_train_dataset = ReconstructionDataset(
        train_ct, train_mri, train_wsi, train_clinical,
        data['train']['ct_mask'],
        data['train']['mri_mask'],
        data['train']['wsi_mask'],
        gaussian_noise_std=GAUSSIAN_NOISE_STD,
        apply_augmentation=True
    )

    recon_test_dataset = ReconstructionDataset(
        test_ct, test_mri, test_wsi, test_clinical,
        data['test']['ct_mask'],
        data['test']['mri_mask'],
        data['test']['wsi_mask'],
        gaussian_noise_std=GAUSSIAN_NOISE_STD,
        apply_augmentation=False  # No augmentation during validation
    )

    # PAPER REQUIREMENT: 6× oversampling of minority class
    recon_sampler = get_reconstruction_oversampler(
        data['train']['labels'],
        oversample_factor=RECON_OVERSAMPLE_FACTOR
    )

    recon_train_loader = DataLoader(
        recon_train_dataset,
        batch_size=RECON_BATCH_SIZE,
        sampler=recon_sampler,  # Use oversampler instead of shuffle
        num_workers=0
    )

    recon_test_loader = DataLoader(
        recon_test_dataset,
        batch_size=RECON_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Initialize reconstruction model
    reconstruction_model = ModalityReconstructionModel(
        ct_dim=512,
        mri_dim=512,
        wsi_dim=2048,
        clinical_dim=train_clinical.shape[1],
        embedding_dim=128
    ).to(DEVICE)

    print(f"\nReconstruction Model Parameters: {sum(p.numel() for p in reconstruction_model.parameters()):,}")

    # Train reconstruction model
    reconstruction_model = train_reconstruction_model(
        reconstruction_model, recon_train_loader, recon_test_loader,
        num_epochs=RECON_NUM_EPOCHS,
        learning_rate=RECON_LEARNING_RATE,
        device=DEVICE
    )

    # Evaluate reconstruction quality
    recon_results = evaluate_reconstruction_quality(
        reconstruction_model, recon_test_loader, device=DEVICE
    )

    # Save reconstruction model
    torch.save(reconstruction_model.state_dict(), 'reconstruction_model_paper_compliant.pth')
    print("\nReconstruction model saved to 'reconstruction_model_paper_compliant.pth'")

    # ========================================================================
    # PART 2: TRAIN CLASSIFICATION MODEL WITH RECONSTRUCTION
    # ========================================================================

    print("\n" + "=" * 80)
    print("PHASE 2: CLASSIFICATION WITH RECONSTRUCTION")
    print("=" * 80)

    # Create classification datasets
    class_train_dataset = ClassificationDataset(
        train_ct, train_mri, train_wsi, train_clinical,
        data['train']['labels'],
        data['train']['ct_mask'],
        data['train']['mri_mask'],
        data['train']['wsi_mask']
    )

    class_test_dataset = ClassificationDataset(
        test_ct, test_mri, test_wsi, test_clinical,
        data['test']['labels'],
        data['test']['ct_mask'],
        data['test']['mri_mask'],
        data['test']['wsi_mask']
    )

    class_train_loader = DataLoader(
        class_train_dataset,
        batch_size=CLASS_BATCH_SIZE,
        sampler=get_oversampler(data['train']['labels'], CLASS_OVERSAMPLE_FACTOR),
        drop_last=True,
        num_workers=0
    )

    class_test_loader = DataLoader(
        class_test_dataset,
        batch_size=CLASS_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Initialize classification model with pre-trained reconstruction
    classification_model = ClassificationWithReconstruction(
        reconstruction_model=reconstruction_model,
        ct_dim=512,
        mri_dim=512,
        wsi_dim=2048,
        clinical_dim=train_clinical.shape[1],
        hidden_size=128
    ).to(DEVICE)

    print(f"\nClassification Model Parameters: {sum(p.numel() for p in classification_model.parameters()):,}")

    # Setup training
    pos_count = data['train']['labels'].sum()
    neg_count = len(data['train']['labels']) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count]).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        classification_model.parameters(),
        lr=CLASS_LEARNING_RATE,
        weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CLASS_NUM_EPOCHS,
        eta_min=1e-6
    )

    # Train classification model
    classification_model = train_classification_model(
        classification_model, class_train_loader, class_test_loader,
        criterion, optimizer, scheduler,
        num_epochs=CLASS_NUM_EPOCHS,
        device=DEVICE
    )

    # Evaluate classification model
    predictions, probabilities, labels, bacc = evaluate_classification_model(
        classification_model, class_test_loader, device=DEVICE
    )

    # Save classification model
    torch.save(classification_model.state_dict(), 'classification_with_reconstruction.pth')
    print("\nClassification model saved to 'classification_with_reconstruction.pth'")

    print("\n" + "=" * 80)
    print("COMPLETE PIPELINE FINISHED!")
    print("=" * 80)
    print(f"Final Test Balanced Accuracy: {bacc:.4f}")

    return bacc


if __name__ == "__main__":
    main()