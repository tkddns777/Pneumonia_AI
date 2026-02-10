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
import random
from torch.utils.data import WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 완전 재현성 (조금 느려질 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =====================================================
# Config (원본 그대로)
# =====================================================
TRAIN_DIR = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia CT images\train"
TEST_DIR  = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia CT images\test"

MODEL_SAVE_DIR = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia models"

BEST_MODEL_PATH = os.path.join(
    MODEL_SAVE_DIR,
    "resnet18_pneumonia_best.pth"
)

BATCH_SIZE = 256
EPOCHS = 5
LR = 2.5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import torch.nn.functional as F

# =====================================================
# Logit 통계 수집 및 추정 함수
@torch.no_grad()
def collect_logits_and_labels(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)              # (B, C) raw logits
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    all_logits = torch.cat(all_logits, dim=0)   # (N, C)
    all_labels = torch.cat(all_labels, dim=0)   # (N,)
    return all_logits, all_labels

# =====================================================
# 클래스별 Logit 통계 추정 함수
def estimate_classwise_logit_stats(logits, labels, num_classes):
    """
    logits: (N, C) CPU tensor
    labels: (N,) CPU tensor (ground-truth)
    return:
      mu:    (C,) 각 클래스의 '정답 클래스 logit' 평균
      sigma: (C,) 각 클래스의 '정답 클래스 logit' 표준편차
    """
    mu = torch.zeros(num_classes, dtype=torch.float32)
    sigma = torch.zeros(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        idx = (labels == c)
        if idx.sum() < 2:
            # 표본이 너무 적으면 분산 추정 불가 -> sigma 아주 작게 처리
            values = logits[idx, c]
            mu[c] = values.mean() if idx.sum() > 0 else 0.0
            sigma[c] = 1e-6
        else:
            values = logits[idx, c]   # "정답 클래스 c" 샘플의 "클래스 c logit"
            mu[c] = values.mean()
            sigma[c] = values.std(unbiased=True) + 1e-6  # 0 방지
    return mu, sigma


def main(seed):
    print(f"\n===== Training with seed = {seed} =====")
    set_seed(seed)
    # =====================================================
    # CUDA print (원본 그대로)
    # =====================================================
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # 폴더 없으면 저장 에러 나니까 안전하게 생성 (기능 추가지만 원본 의도 유지)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # =====================================================
    # train transform with data augmentation (원본 그대로)
    # =====================================================
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # 위치/크기 다양화
        transforms.RandomRotation(15),                        # 촬영 각도 편차
        # transforms.RandomHorizontalFlip(p=0.5),              # 좌우 대칭 허용 (흉부는 OK)
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # 촬영 조건 차이
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # =====================================================
    # Transform (test) (원본 그대로)
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
    # Dataset & Loader (원본 유지 + Windows 안정화 옵션만 추가)
    # =====================================================
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset  = datasets.ImageFolder(TEST_DIR, transform=transform)

    class_names = train_dataset.classes
    print("Classes:", class_names)

    # ✅ 여기서 num_workers가 0이 아니면 Windows에서 __main__ 가드 없을 때 터짐
    # 지금은 __main__ 가드가 있으니 num_workers 사용 가능
    # 다만, VSCode/환경 따라 문제 생기면 num_workers=0으로 낮추면 됨
    # 불균형 데이터셋 처리를 위한 WeightedRandomSampler 추가
    targets = [y for _, y in train_dataset.samples]      # ImageFolder의 라벨 리스트
    class_counts = np.bincount(targets)                 # 클래스별 개수
    print("Train class counts:", class_counts)

    # 클래스가 적을수록 더 많이 뽑히도록 가중치 부여
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),   # 한 epoch당 샘플 수 (보통 전체 길이)
        replacement=True                   # 소수 클래스는 중복 허용
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,                   # ✅ shuffle 대신 sampler 사용
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=torch.cuda.is_available())

    # =====================================================
    # Model (원본 그대로)
    # =====================================================
    model = models.resnet18(pretrained=True)

    model.fc = nn.Sequential(
        # nn.Dropout(p=0.5),  # 필요시 여기에 활성화
        nn.Linear(model.fc.in_features, len(class_names))
    )

    model.to(DEVICE)

    # =====================================================
    # Loss & Optimizer (원본 그대로)
    # =====================================================
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # =====================================================
    # Training + Evaluation (원본 그대로)
    # =====================================================
    best_test_acc = 0.0
    for epoch in range(EPOCHS):
        # ------------------
        # Train
        # ------------------,
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
        # Train Confusion Matrix (원본 그대로)
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
        plt.pause(0.5)
        plt.close()

        # ------------------
        # Test
        # ------------------
        # ------------------
        # Test (logits 기반 gate 포함)
        # ------------------
        model.eval()

        # 1) test logits/labels 수집
        test_logits, test_labels_t = collect_logits_and_labels(model, test_loader, DEVICE)

        # 2) 일반 예측(기존 방식)
        test_preds_t = test_logits.argmax(dim=1)
        test_acc = (test_preds_t == test_labels_t).float().mean().item()
        print(f"[Test] Accuracy: {test_acc:.3f}")

        # 3) 클래스별 logit 분포(μ, σ) 추정 (정답 클래스 logit 기준)
        num_classes = len(class_names)
        mu, sigma = estimate_classwise_logit_stats(test_logits, test_labels_t, num_classes)

        # 4) gate 정의: "예측 클래스의 logit"이 (μ_pred + k·σ_pred) 이상이면 confident
        k = 1.0  # 보통 0.5~2.0 사이에서 튜닝 (1.0부터 추천)
        pred_cls = test_preds_t
        pred_logit = test_logits[torch.arange(test_logits.size(0)), pred_cls]
        thresholds = mu[pred_cls] + k * sigma[pred_cls]
        confident_mask = pred_logit >= thresholds

        coverage = confident_mask.float().mean().item()  # confident로 남는 비율
        if confident_mask.sum() > 0:
            confident_acc = (test_preds_t[confident_mask] == test_labels_t[confident_mask]).float().mean().item()
        else:
            confident_acc = 0.0

        print(f"[Gate] k={k:.2f} | Coverage: {coverage:.3f} | Confident Acc: {confident_acc:.3f}")

        # (선택) gate 통과한 샘플만 confusion matrix
        # 단, coverage가 너무 낮으면 의미가 없을 수 있음
        if confident_mask.sum() > 0:
            cm_test_conf = confusion_matrix(
                test_labels_t[confident_mask].numpy(),
                test_preds_t[confident_mask].numpy()
            )
            plt.figure(figsize=(4, 4))
            sns.heatmap(
                cm_test_conf, annot=True, fmt="d", cmap="Oranges",
                xticklabels=class_names, yticklabels=class_names
            )
            plt.title(f"Confident Test CM (k={k:.1f}, cov={coverage:.2f})")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()
        

        # ------------------
        # Save best model (원본 그대로 + model_name만 정정)
        # ------------------
        SAVE_THRESHOLD = 0.95
        MIN_COVERAGE = 0.40
        MIN_CONF_ACC = 0.97

        if (test_acc >= SAVE_THRESHOLD) and (coverage >= MIN_COVERAGE) and (confident_acc >= MIN_CONF_ACC):
            save_path = os.path.join(
                MODEL_SAVE_DIR,
                f"resnet18_seed{seed}_epoch{epoch+1:03d}_acc{test_acc:.3f}_cov{coverage:.2f}_cacc{confident_acc:.3f}.pth"
            )

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_accuracy": test_acc,
                "gate_k": k,
                "gate_coverage": coverage,
                "gate_confident_accuracy": confident_acc,
                "class_names": class_names,
                "model_name": "resnet18",
                "logit_mu": mu.numpy(),       # 나중에 재현 가능
                "logit_sigma": sigma.numpy(), # 나중에 재현 가능
            }, save_path)

            print(f"✅ Saved (acc={test_acc:.3f}, cov={coverage:.2f}, cacc={confident_acc:.3f})")


    # =====================================================
    # Save Model (원본 그대로)
    # =====================================================
    torch.save(model.state_dict(), "model.pth")
    print("Model saved.")


# ✅ Windows 멀티프로세싱 필수 가드
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    SEEDS = [0, 100, 500]   # 추천: 3개면 충분

    for seed in SEEDS:
        main(seed)