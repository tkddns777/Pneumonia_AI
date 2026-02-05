import os, glob
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

LOGIT_CLIP = 5.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SAVE_DIR = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia models"
TEST_DIR = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia CT images\test"

def build_model(num_classes: int):
    m = models.resnet18(weights=None)  # pretrained=False 대체 (경고 없음)
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, num_classes))  # 학습과 동일
    return m

def main():
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # 1) 모델 파일들 가져오기
    model_paths = sorted(glob.glob(os.path.join(MODEL_SAVE_DIR, "resnet18_epoch*_acc*.pth")))
    if len(model_paths) == 0:
        raise FileNotFoundError("No saved models found. Check MODEL_SAVE_DIR and filename pattern.")

    print("Found models:", len(model_paths))
    for p in model_paths:
        print(" -", os.path.basename(p))

    # 2) 테스트 transform (훈련의 test와 맞춤)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

    # ✅ Windows 안정: num_workers>0 사용하려면 main 가드가 반드시 필요(우린 이미 main 안)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # 3) 첫 ckpt에서 class_names 가져오기
    first_ckpt = torch.load(model_paths[0], map_location="cpu")
    class_names = first_ckpt["class_names"]
    num_classes = len(class_names)

    print("Checkpoint class_names:", class_names)
    print("Test dataset classes:", test_dataset.classes)

    if test_dataset.classes != class_names:
        print("⚠️ WARNING: test_dataset.classes != checkpoint class_names")
        print("=> 라벨 매핑 문제 가능. 폴더명/정렬 확인 필요.")

    # 4) GT 라벨(고정)
    all_labels = []
    for _, labels in test_loader:
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels, axis=0)

    # 5) logits 누적 버퍼
    N = len(test_dataset)
    sum_logits = np.zeros((N, num_classes), dtype=np.float32)

    # 6) 모델별 순차 평가 + logits 누적
    with torch.no_grad():
        for i, mp in enumerate(model_paths, start=1):
            ckpt = torch.load(mp, map_location="cpu")

            model = build_model(num_classes)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model.to(DEVICE)
            model.eval()

            logits_list = []
            for images, _ in test_loader:
                images = images.to(DEVICE, non_blocking=True)
                out = model(images)  # logits
                logits_list.append(out.detach().cpu().numpy())

            logits = np.concatenate(logits_list, axis=0)
            logits = np.clip(logits, -LOGIT_CLIP, LOGIT_CLIP)
            sum_logits += logits

            # 개별 모델 성능도 같이 출력
            preds_i = logits.argmax(axis=1)
            acc_i = accuracy_score(all_labels, preds_i)
            print(f"[{i}/{len(model_paths)}] {os.path.basename(mp)} | acc={acc_i:.3f}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 7) 앙상블 (avg logits)
    avg_logits = sum_logits / len(model_paths)
    ens_preds = avg_logits.argmax(axis=1)

    ens_acc = accuracy_score(all_labels, ens_preds)
    ens_cm = confusion_matrix(all_labels, ens_preds)

    print(f"\n✅ ENSEMBLE accuracy (avg logits): {ens_acc:.3f}")

    # 클래스별 recall도 같이(체감 이상함 확인용)
    tn, fp, fn, tp = ens_cm.ravel()
    normal_recall = tn / (tn + fp + 1e-9)
    pneu_recall = tp / (tp + fn + 1e-9)
    print(f"NORMAL recall   : {normal_recall:.3f}")
    print(f"PNEUMONIA recall: {pneu_recall:.3f}")

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        ens_cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Ensemble Confusion Matrix (avg logits)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
