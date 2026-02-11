import os, glob
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SAVE_DIR = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia models"
TEST_DIR = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia CT images\test"

def build_model(num_classes: int):
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, num_classes))  # 학습과 동일
    return m

def main():
    # 1) 모델 파일들 가져오기
    model_paths = sorted(glob.glob(os.path.join(MODEL_SAVE_DIR, "resnet18_seed*_epoch*_acc*.pth")))
    if len(model_paths) == 0:
        raise FileNotFoundError("No saved models found. Check MODEL_SAVE_DIR and filename pattern.")

    print("Found models:", len(model_paths))
    for p in model_paths:
        print(" -", os.path.basename(p))

    # 2) 테스트 transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
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

    # 5) 모델별 예측을 모을 버퍼: (num_models, N)
    N = len(test_dataset)
    all_model_preds = np.zeros((len(model_paths), N), dtype=np.int64)

    # (선택) 동률 깨기용: 평균 softmax 확률 누적 (안전장치)
    # 동률이 없으면 사실상 안 쓰임.
    sum_probs = np.zeros((N, num_classes), dtype=np.float32)

    softmax = torch.nn.Softmax(dim=1)

    # 6) 모델별 순차 평가
    with torch.no_grad():
        for i, mp in enumerate(model_paths, start=1):
            ckpt = torch.load(mp, map_location="cpu")

            model = build_model(num_classes)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model.to(DEVICE)
            model.eval()

            preds_list = []
            probs_list = []

            for images, _ in test_loader:
                images = images.to(DEVICE, non_blocking=True)
                out = model(images)  # logits
                preds = out.argmax(dim=1).detach().cpu().numpy()
                preds_list.append(preds)

                # 동률 깨기용(평균 확률)
                probs = softmax(out).detach().cpu().numpy()
                probs_list.append(probs)

            preds_i = np.concatenate(preds_list, axis=0)
            probs_i = np.concatenate(probs_list, axis=0)

            all_model_preds[i-1, :] = preds_i
            sum_probs += probs_i

            # 개별 모델 성능 출력
            acc_i = accuracy_score(all_labels, preds_i)
            print(f"[{i}/{len(model_paths)}] {os.path.basename(mp)} | acc={acc_i:.3f}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 7) 다수결(majority vote)
    # 각 샘플별로 모델들이 낸 예측들의 최빈값을 고름
    ens_preds = np.zeros((N,), dtype=np.int64)

    for j in range(N):
        votes = all_model_preds[:, j]               # shape: (num_models,)
        counts = np.bincount(votes, minlength=num_classes)
        top = counts.max()
        winners = np.where(counts == top)[0]

        if len(winners) == 1:
            ens_preds[j] = winners[0]
        else:
            # 동률이면 평균 확률이 큰 쪽으로 결정(안전장치)
            avg_probs = sum_probs[j] / len(model_paths)
            ens_preds[j] = int(np.argmax(avg_probs))

    ens_acc = accuracy_score(all_labels, ens_preds)
    ens_cm = confusion_matrix(all_labels, ens_preds)

    print(f"\n✅ ENSEMBLE accuracy (majority vote): {ens_acc:.3f}")

    # 클래스별 recall (2-class 가정이면 ravel 가능)
    if ens_cm.shape == (2, 2):
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
    plt.title("Ensemble Confusion Matrix (majority vote)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
