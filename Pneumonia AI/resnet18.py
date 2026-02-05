import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix


print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# =====================================================
# Config
# =====================================================
TRAIN_DIR = r"/content/data/train"
TEST_DIR  = r"/content/data/test"

MODEL_SAVE_DIR = r"/content/results"

BEST_MODEL_PATH = os.path.join(
    MODEL_SAVE_DIR,
    "resnet18_pneumonia_best.pth"
)

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# train transform with data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # 위치/크기 다양화
    transforms.RandomRotation(15),                        # 촬영 각도 편차
    #transforms.RandomHorizontalFlip(p=0.5),               # 좌우 대칭 허용 (흉부는 OK)
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # 촬영 조건 차이
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# Transform
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# Dataset & Loader
# =====================================================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset  = datasets.ImageFolder(TEST_DIR, transform=transform)

class_names = train_dataset.classes
print("Classes:", class_names)

# ✅ [1번 적용] DataLoader 튜닝만 추가
NUM_WORKERS = min(16, os.cpu_count() or 8)  # Colab 환경에서 보통 2~8이 무난
PREFETCH = 4

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True if NUM_WORKERS > 0 else False,
    prefetch_factor=PREFETCH
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True if NUM_WORKERS > 0 else False,
    prefetch_factor=PREFETCH
)

# =====================================================
# Model
# =====================================================
model = models.resnet18(pretrained=True)

model.fc = nn.Sequential(
    #nn.Dropout(p=0.5),  # 필요시 여기에 활성화
    nn.Linear(model.fc.in_features, len(class_names))
)

model.to(DEVICE)

# =====================================================
# Loss & Optimizer
# =====================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =====================================================
# Training + Evaluation
# =====================================================
best_test_acc = 0.0
for epoch in range(EPOCHS):
    # ------------------
    # Train
    # ------------------
    model.train()
    train_preds, train_labels = [], []
    running_loss = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_preds.extend(outputs.argmax(1).cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)
    print(f"[Train] Loss: {running_loss/len(train_loader):.4f} | Acc: {train_acc:.3f}")

    # ------------------
    # Train Confusion Matrix
    # ------------------
    cm_train = confusion_matrix(train_labels, train_preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm_train, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(f"Train Confusion Matrix (Epoch {epoch+1})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    # ------------------
    # Test
    # ------------------
    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1} [Test]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            test_preds.extend(outputs.argmax(1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    print(f"[Test] Accuracy: {test_acc:.3f}")

    # ------------------
    # Test Confusion Matrix
    # ------------------
    cm_test = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm_test, annot=True, fmt="d", cmap="Greens",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)

    # ------------------
    # Save best model
    # ------------------
    if test_acc > best_test_acc:
        best_test_acc = test_acc

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_accuracy": test_acc,
            "class_names": class_names,
            "model_name": "resnet50",
        }, BEST_MODEL_PATH)

        print(f"✅ Best model saved (epoch={epoch+1}, acc={test_acc:.3f})")


# =====================================================
# Save Model
# =====================================================
torch.save(model.state_dict(), "model.pth")
print("Model saved.")
