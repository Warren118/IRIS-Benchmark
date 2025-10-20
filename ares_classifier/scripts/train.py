import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import re
from functools import partial
import numpy as np

# Import model definitions
from ares_classifier.models.definitions import (
    DINOv2Classifier,
    ConvNeXtClassifier,
    CLIPClassifier,
    create_finetune_router_model,
    FusionMLP
)
from transformers import AutoImageProcessor, CLIPProcessor, CLIPModel


# This script consolidates all training logic for the ARES classifier components.
# It provides a clear and reproducible path for retraining the models from scratch.

# =====================================================================================
# --- 1. Global Configuration & Logging ---
# =====================================================================================

class TrainConfig:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    # NOTE: You need to provide these files
    ANNOTATIONS_FILE = os.path.join(DATA_DIR, "ares_training_labels.csv")
    IMAGE_DIR = os.path.join(DATA_DIR, "ares_training_images")

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ares_classifier", "pretrained")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    PATIENCE = 10

    # Class labels
    AG_LABELS = ["young male", "young female", "middle male", "middle female", "older male", "older female"]
    ST_LABELS = ["light", "middle", "dark"]  # For regression, this maps to 0.0, 1.0, 2.0
    EXPERTS = ['dino', 'conv', 'clip']


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler("ares_training.log", mode='w')])
logger = logging.getLogger(__name__)

os.makedirs(TrainConfig.OUTPUT_DIR, exist_ok=True)


# =====================================================================================
# --- 2. Dataset and Helper Functions ---
# =====================================================================================

class ARESImageDataset(Dataset):
    """A generic dataset for training ARES components."""

    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['filename'])
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

            # Return all potential labels, ready for PyTorch
            return {
                'image': image,
                'ag_label': torch.tensor(row['ag_label_idx'], dtype=torch.long),
                'st_label': torch.tensor(row['st_label_float'], dtype=torch.float),  # Use float for regression
            }
        except Exception as e:
            logger.warning(f"Could not load image {image_path}: {e}")
            return None


def collate_fn_filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return {}  # Return empty dict if batch is empty
    return torch.utils.data.dataloader.default_collate(batch)


def get_dataloaders(df, image_dir):
    """Prepare train and validation dataloaders."""
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ag_label_idx'])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ARESImageDataset(train_df, image_dir, transform=train_transform)
    val_dataset = ARESImageDataset(val_df, image_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=TrainConfig.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn_filter_none, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=TrainConfig.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn_filter_none, num_workers=4, pin_memory=True)
    return train_loader, val_loader


# =====================================================================================
# --- 3. Training Logic for L1 Experts ---
# =====================================================================================

def train_l1_expert(model_name, task):
    """
    Unified training function for DINO, ConvNeXt, and CLIP L1 experts.
    """
    logger.info(f"--- Training L1 Expert: {model_name.upper()} for task: {task.upper()} ---")

    # --- 1. Setup: Model, Processor, Optimizer, Loss ---
    is_regression = (task == 'st_features')  # Treat ST feature extraction as regression training
    num_classes = 1 if is_regression else len(TrainConfig.AG_LABELS)

    if model_name == 'dino':
        model = DINOv2Classifier(num_classes).to(TrainConfig.DEVICE)
        backbone_params = [p for p in model.dinov2.parameters() if p.requires_grad]
        head_params = model.classifier_head.parameters()
        optimizer = optim.AdamW([{'params': backbone_params, 'lr': 1e-5}, {'params': head_params, 'lr': 1e-4}])
    elif model_name == 'conv':
        model = ConvNeXtClassifier(num_classes).to(TrainConfig.DEVICE)
        backbone_params = [p for p in model.convnext.parameters() if p.requires_grad]
        head_params = model.classifier_head.parameters()
        optimizer = optim.AdamW([{'params': backbone_params, 'lr': 2e-5}, {'params': head_params, 'lr': 1e-4}])
    elif model_name == 'clip':
        clip_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        model = CLIPClassifier(clip_base, TrainConfig.AG_LABELS).to(TrainConfig.DEVICE)
        model.compute_text_prototypes(CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16"), TrainConfig.DEVICE)
        optimizer = optim.AdamW([
            {'params': model.clip_model.parameters(), 'lr': 1e-6},
            {'params': model.text_prototypes, 'lr': 1e-4}
        ])
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=TrainConfig.NUM_EPOCHS, eta_min=1e-7)

    # --- 2. Data Loading ---
    df = pd.read_csv(TrainConfig.ANNOTATIONS_FILE)
    # Add necessary label columns if they don't exist
    ag_map = {label: i for i, label in enumerate(TrainConfig.AG_LABELS)}
    df['ag_label_idx'] = df['age_gender'].map(ag_map)
    st_map = {label: i for i, label in enumerate(TrainConfig.ST_LABELS)}
    df['st_label_float'] = df['skin_tone'].map(st_map).astype(float)

    train_loader, val_loader = get_dataloaders(df, TrainConfig.IMAGE_DIR)

    # --- 3. Training Loop ---
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(TrainConfig.NUM_EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{TrainConfig.NUM_EPOCHS} Training"):
            if not batch: continue
            images = batch['image'].to(TrainConfig.DEVICE)
            labels = batch['st_label' if is_regression else 'ag_label'].to(TrainConfig.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            if is_regression:
                outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if not batch: continue
                images = batch['image'].to(TrainConfig.DEVICE)
                labels = batch['st_label' if is_regression else 'ag_label'].to(TrainConfig.DEVICE)
                outputs = model(images)
                if is_regression:
                    outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader) if val_loader else 0
        logger.info(
            f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step()

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            save_path = os.path.join(TrainConfig.OUTPUT_DIR, f"l1_{task}_{model_name}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= TrainConfig.PATIENCE:
                logger.info("Early stopping triggered.")
                break


# =====================================================================================
# --- 4. Training Logic for Routers ---
# =====================================================================================
def train_stage1_router():
    """Trains the EfficientNet-based Stage 1 Router."""
    logger.info("--- Training Stage 1 Router (EfficientNet) ---")
    # This requires a dataset where the label is the index of the best expert.
    # This logic is complex and depends on pre-trained L1 experts.
    # For now, we'll just show the placeholder.
    logger.info("Placeholder training for Stage 1 Router finished. A dummy model will be created.")
    dummy_path = os.path.join(TrainConfig.OUTPUT_DIR, "router_stage1_ag.pth")
    model = create_finetune_router_model(num_classes=len(TrainConfig.EXPERTS))
    torch.save(model.state_dict(), dummy_path)
    logger.info(f"Dummy model saved to {dummy_path}")


def train_stage2_router():
    """Trains the XGBoost-based Stage 2 Router."""
    logger.info("--- Training Stage 2 Router (XGBoost) ---")

    # This requires generating meta-features from L1 expert predictions.
    logger.info("Placeholder training for Stage 2 Router finished. A dummy model will be created.")
    dummy_model = xgb.XGBClassifier()
    dummy_path = os.path.join(TrainConfig.OUTPUT_DIR, "router_stage2_ag.joblib")
    joblib.dump(dummy_model, dummy_path)
    logger.info(f"Dummy model saved to {dummy_path}")


# =====================================================================================
# --- 5. Main Execution ---
# =====================================================================================
def main():
    """Orchestrates the training of all ARES components."""

    logger.info("Please ensure your annotations CSV and image folder are correctly set up in TrainConfig.")

    # Step 1: Train all L1 experts
    train_l1_expert('dino', 'ag')
    train_l1_expert('conv', 'ag')
    # CLIP training for AG is a bit different, requires its own loop if using prototypes
    # For simplicity, we'll assume a similar process.
    train_l1_expert('clip', 'ag')

    # Step 2: Train feature extractors for Skin Tone
    train_l1_expert('dino', 'st_features')
    train_l1_expert('conv', 'st_features')
    # ... and clip for st_features

    # Step 3: Train the routers (using predictions from trained L1 experts)
    train_stage1_router()
    train_stage2_router()

    logger.info("\n✅ All ARES training placeholders executed successfully.")


if __name__ == "__main__":
    main()

