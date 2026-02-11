import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from io import BytesIO
from pathlib import Path
import seaborn as sns

import matplotlib.pyplot as plt
from PIL import Image

from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# EfficientNet (optional)
from torchvision.models import efficientnet_b0

# =====================================================
# 설정: 너 환경에 맞게 여기만 수정
# =====================================================
MODEL_PATH = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia models\Inception_V3_seed0_epoch004_acc0.965.pth"
TEST_DIR   = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia CT images\test"   # test/NORMAL, test/PNEUMONIA
OUT_DIR    = r"C:\Users\user\OneDrive\바탕 화면\코딩 연습\Pneumonia AI\Metrics_results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

# =====================================================
# Unicode-safe 저장 (한글/OneDrive/공백 경로에서도 OK)
# =====================================================
def imwrite_unicode(path, img_bgr_or_bgra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext == "":
        ext = ".png"
        path = path.with_suffix(ext)

    img = np.ascontiguousarray(img_bgr_or_bgra)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for ext={ext}")

    buf.tofile(str(path))

    if (not path.exists()) or path.stat().st_size == 0:
        raise RuntimeError(f"File not created or empty: {path}")

    return str(path.resolve())

def savefig_unicode(fig, save_path, dpi=200):
    """
    matplotlib figure를 한글/공백 경로에서도 안전하게 저장
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)

    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("cv2.imdecode failed while saving figure")

    saved = imwrite_unicode(str(save_path), img)
    plt.close(fig)
    return saved

# =====================================================
# 모델 로드: ResNet18 / EfficientNet-b0 / InceptionV3 지원
# =====================================================
def build_model_from_checkpoint(checkpoint, device):
    model_name = checkpoint.get("model_name", "resnet18")
    class_names = checkpoint.get("class_names", ["NORMAL", "PNEUMONIA"])
    num_classes = len(class_names)

    mn = model_name.lower().replace("-", "_")

    # -------- InceptionV3 --------
    if mn in ["inception_v3", "inceptionv3", "inception"]:
        model = models.inception_v3(weights=None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        return model.to(device), class_names, mean, std, model_name

    # -------- EfficientNet-b0 --------
    if mn in ["efficientnet_b0", "efficientnet"]:
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        return model.to(device), class_names, mean, std, model_name

    # -------- ResNet18 --------
    if mn in ["resnet18", "resnet_18"]:
        model = models.resnet18(weights=None)

        state_dict = checkpoint["model_state_dict"]
        # dropout 포함 저장된 경우 대응
        if any(k.startswith("fc.1.") for k in state_dict.keys()):
            model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(model.fc.in_features, num_classes)
            )
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        return model.to(device), class_names, mean, std, model_name

    raise RuntimeError(f"지원하지 않는 model_name='{model_name}'. checkpoint['model_name'] 확인 필요")

# =====================================================
# Confusion Matrix figure 생성/표시/저장
# =====================================================
def plot_confusion_matrix(cm, class_names, title, normalize=False, cmap="Blues"):
    """
    디자인: 사용자가 학습 코드에서 쓰던 sns.heatmap 스타일 참고
    - raw: 정수 count 표시
    - normalize: 행 기준(=True class 기준) 퍼센트 표시
    """
    if normalize:
        cm_plot = cm.astype(np.float64)
        cm_plot = cm_plot / (cm_plot.sum(axis=1, keepdims=True) + 1e-12)

        # 퍼센트 텍스트로 annot 만들기
        annot = np.array([[f"{v*100:.1f}%" for v in row] for row in cm_plot])
        fmt = ""  # annot를 문자열로 넣으니 fmt 비움
        data = cm_plot
    else:
        annot = True
        fmt = "d"
        data = cm

    fig = plt.figure(figsize=(4.5, 4.2))  # 네가 쓰던 4x4 느낌
    ax = fig.add_subplot(111)

    sns.heatmap(
        data,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,            # ✅ 정사각형 셀
        linewidths=0.5,         # ✅ 경계선 약간
        linecolor="white",
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    return fig

# =====================================================
# Metrics table figure 생성/표시/저장
# =====================================================
def plot_metrics_table(class_names, precision, recall, f1, support,
                       acc, macro_p, macro_r, macro_f1, weighted_f1):
    col_labels = ["Class", "Precision", "Recall", "F1", "Support"]
    rows = []
    for i, cname in enumerate(class_names):
        rows.append([
            cname,
            f"{precision[i]:.3f}",
            f"{recall[i]:.3f}",
            f"{f1[i]:.3f}",
            f"{int(support[i])}"
        ])

    rows.append(["", "", "", "", ""])
    rows.append(["Accuracy", "", "", f"{acc:.3f}", f"{int(np.sum(support))}"])
    rows.append(["Macro avg", f"{macro_p:.3f}", f"{macro_r:.3f}", f"{macro_f1:.3f}", f"{int(np.sum(support))}"])
    rows.append(["Weighted avg", "", "", f"{weighted_f1:.3f}", f"{int(np.sum(support))}"])

    fig = plt.figure(figsize=(9, 0.5 + 0.35 * (len(rows) + 1)))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title("Test set metrics", pad=12)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)

    fig.tight_layout()
    return fig

# =====================================================
# TEST 전체 평가
# =====================================================
@torch.inference_mode()
def evaluate_testset(model, test_dir, preprocess, class_names, device):
    dataset = ImageFolder(root=test_dir, transform=preprocess)

    # ImageFolder 클래스 순서가 checkpoint class_names와 다를 수 있어 안전 처리
    ds_classes = dataset.classes
    if list(ds_classes) != list(class_names):
        print(f"[WARN] dataset.classes={ds_classes} != checkpoint class_names={class_names}")
        print("[WARN] class_names 기준으로 라벨 재정렬해서 평가합니다.")
        ds_idx_to_name = {i: n for i, n in enumerate(ds_classes)}
        desired_name_to_idx = {n: i for i, n in enumerate(class_names)}
        remap = {ds_i: desired_name_to_idx[ds_idx_to_name[ds_i]] for ds_i in range(len(ds_classes))}
    else:
        remap = None

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,  # ✅ Windows 안정
        pin_memory=(device == "cuda")
    )

    y_true_list, y_pred_list = [], []

    model.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)

        # inception에서 aux logits tuple 방어
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_np = y.cpu().numpy()

        if remap is not None:
            y_np = np.array([remap[int(v)] for v in y_np], dtype=np.int64)

        y_true_list.append(y_np)
        y_pred_list.append(pred)

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # metrics
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), average="macro", zero_division=0
    )

    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), average="weighted", zero_division=0
    )

    return cm, acc, precision, recall, f1, support, macro_p, macro_r, macro_f1, weighted_f1

# =====================================================
# MAIN
# =====================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_dir = Path(OUT_DIR)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1) checkpoint load
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"]

    # 2) build model
    model, class_names, mean, std, model_name = build_model_from_checkpoint(checkpoint, DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"[INFO] Loaded model_name: {model_name}")
    print(f"[INFO] Classes: {class_names}")

    # 3) preprocess
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # 4) evaluate full test set
    cm, acc, precision, recall, f1, support, macro_p, macro_r, macro_f1, weighted_f1 = evaluate_testset(
        model=model,
        test_dir=TEST_DIR,
        preprocess=preprocess,
        class_names=class_names,
        device=DEVICE
    )

    # ===== print metrics (콘솔 출력)
    print("\n==============================")
    print("[TEST METRICS]")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Macro Precision : {macro_p:.4f}")
    print(f"Macro Recall    : {macro_r:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print(f"Weighted F1     : {weighted_f1:.4f}")
    print("------------------------------")
    for i, cname in enumerate(class_names):
        print(f"{cname:12s}  P={precision[i]:.4f}  R={recall[i]:.4f}  F1={f1[i]:.4f}  Support={int(support[i])}")
    print("==============================\n")

    # 5) Figures 생성 (보여주고 + 저장)
    fig_cm_raw = plot_confusion_matrix(
        cm, class_names,
        title=f"Confusion Matrix (raw) | Acc={acc:.3f}",
        normalize=False
    )
    plt.show()
    save_raw = savefig_unicode(fig_cm_raw, metrics_dir / "cm_raw.png")
    print(f"[OK] Saved: {save_raw}")

    fig_cm_norm = plot_confusion_matrix(
        cm, class_names,
        title=f"Confusion Matrix (normalized) | Acc={acc:.3f}",
        normalize=True
    )
    plt.show()
    save_norm = savefig_unicode(fig_cm_norm, metrics_dir / "cm_norm.png")
    print(f"[OK] Saved: {save_norm}")

    fig_table = plot_metrics_table(
        class_names, precision, recall, f1, support,
        acc, macro_p, macro_r, macro_f1, weighted_f1
    )
    plt.show()
    save_table = savefig_unicode(fig_table, metrics_dir / "metrics.png")
    print(f"[OK] Saved: {save_table}")

    print("\n[DONE] Saved all figures to:", str(metrics_dir))
    print(" - cm_raw.png")
    print(" - cm_norm.png")
    print(" - metrics.png")

if __name__ == "__main__":
    main()
