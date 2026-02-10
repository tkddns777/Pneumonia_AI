import os
import glob
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt
from pathlib import Path
import random

# =====================================================
# 설정: 너 환경에 맞게 여기만 수정
# =====================================================
MODEL_PATH = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia models\Inception_V3_seed0_epoch004_acc0.965.pth"
TEST_DIR   = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia CT images\test"   # test/NORMAL, test/PNEUMONIA
OUT_DIR    = r"C:\Users\user\OneDrive\바탕 화면\코딩 연습\Pneumonia AI\Grad-CAM\gradcam_test2"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 모델 학습 시 사용한 이미지 크기와 동일하게!
Image_SIZE = 224


# =====================================================
# Unicode-safe 저장 (한글/OneDrive/공백 경로에서도 OK)
# =====================================================
def imwrite_unicode(path, img_bgr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext == "":
        ext = ".png"
        path = path.with_suffix(ext)

    img_bgr = np.ascontiguousarray(img_bgr)
    if img_bgr.dtype != np.uint8:
        img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for ext={ext}")

    buf.tofile(str(path))

    if (not path.exists()) or path.stat().st_size == 0:
        raise RuntimeError(f"File not created or empty: {path}")

    return str(path.resolve())


# =====================================================
# Grad-CAM (hook 기반)
# =====================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fh = self.target_layer.register_forward_hook(self._forward_hook)
        self.bh = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self):
        self.fh.remove()
        self.bh.remove()

    def __call__(self, input_tensor, target_class_idx=None):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)  # (1,num_classes)
        probs = torch.softmax(logits, dim=1)

        pred_idx = int(torch.argmax(probs, dim=1).item())
        pred_prob = float(probs[0, pred_idx].item())

        if target_class_idx is None:
            target_class_idx = pred_idx

        score = logits[0, target_class_idx]
        score.backward(retain_graph=False)

        grads = self.gradients            # (1,C,H,W)
        acts = self.activations           # (1,C,H,W)
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam = torch.sum(weights * acts, dim=1)                 # (1,H,W)
        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()[0]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam, pred_idx, pred_prob


# =====================================================
# 유틸: 테스트 폴더에서 클래스별로 이미지 1장 고르기
# =====================================================
def find_one_image_in_folder(folder_path):
    if not os.path.isdir(folder_path):
        return None
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    files = list(set(files))
    if len(files) == 0:
        return None
    return random.choice(files)


def pick_test_images(test_dir, class_names):
    picks = {}
    subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    subdirs_lower_map = {d.lower(): d for d in subdirs}

    for cname in class_names:
        direct = os.path.join(test_dir, cname)
        img = find_one_image_in_folder(direct)
        if img is not None:
            picks[cname] = img
            continue

        key = cname.lower()
        if key in subdirs_lower_map:
            folder = os.path.join(test_dir, subdirs_lower_map[key])
            picks[cname] = find_one_image_in_folder(folder)
        else:
            picks[cname] = None

    return picks


# =====================================================
# Grad-CAM 오버레이 출력 + 저장
# =====================================================
def overlay_and_save(orig_rgb_np, cam_01, save_path, text=None, alpha=0.40, show=True):
    h, w = orig_rgb_np.shape[:2]
    cam_resized = cv2.resize(cam_01, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap = np.uint8(np.clip(cam_resized * 255.0, 0, 255))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR

    if orig_rgb_np.dtype != np.uint8:
        orig_rgb_np = np.clip(orig_rgb_np, 0, 255).astype(np.uint8)
    orig_bgr = cv2.cvtColor(orig_rgb_np, cv2.COLOR_RGB2BGR)

    overlay_bgr = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap, alpha, 0)

    if text:
        cv2.putText(
            overlay_bgr, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255),
            2, cv2.LINE_AA
        )

    if show:
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(7, 7))
        plt.imshow(overlay_rgb)
        plt.axis("off")
        plt.title(text if text else "Grad-CAM Overlay")
        plt.tight_layout()
        plt.show()

    saved_path = imwrite_unicode(save_path, overlay_bgr)
    print(f"[OK] Saved (unicode-safe): {saved_path}")
    return saved_path


# =====================================================
# 모델 로드 + 타겟 레이어 자동 선택
# =====================================================
def build_model_from_checkpoint(checkpoint, device):
    """
    checkpoint에 저장된 model_name에 따라 모델을 구성하고 state_dict 로드.
    """
    model_name = checkpoint.get("model_name", "resnet18")
    class_names = checkpoint.get("class_names", ["NORMAL", "PNEUMONIA"])
    num_classes = len(class_names)

    # ------------------------
    # EfficientNet-b0
    # ------------------------
    if model_name.lower() in ["efficientnet_b0", "efficientnet-b0", "efficientnet"]:
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=None)  # 체크포인트 로드할 거니까 weights=None

        # classifier 마지막 Linear 교체
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

        target_layer = model.features[-1]  # ✅ 마지막 conv block 쪽

        normalize_mean = weights.transforms().mean
        normalize_std = weights.transforms().std

        return model.to(device), target_layer, class_names, normalize_mean, normalize_std, model_name

    # ------------------------
    # ResNet18 (기존 호환)
    # ------------------------
    elif model_name.lower() in ["resnet18", "resnet-18"]:
        model = models.resnet18(weights=None)

        state_dict = checkpoint["model_state_dict"]
        # fc 구조 맞추기
        if any(k.startswith("fc.1.") for k in state_dict.keys()):
            model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(model.fc.in_features, num_classes)
            )
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        target_layer = model.layer4[-1]

        # ImageNet 기본값
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]

        return model.to(device), target_layer, class_names, normalize_mean, normalize_std, model_name

    else:
        raise RuntimeError(f"지원하지 않는 model_name='{model_name}'. checkpoint['model_name']를 확인하세요.")


def main():
    # -----------------------
    # 1) 체크포인트 로드
    # -----------------------
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"]

    # -----------------------
    # 2) 모델 구성 (resnet/efficientnet 자동)
    # -----------------------
    model, target_layer, class_names, mean, std, model_name = build_model_from_checkpoint(checkpoint, DEVICE)

    # strict 로드
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"[INFO] Loaded model_name = {model_name}")
    print(f"[INFO] Classes = {class_names}")

    # -----------------------
    # 3) 전처리 (학습과 동일해야 정확)
    # -----------------------
    preprocess = transforms.Compose([
        transforms.Resize((Image_SIZE, Image_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # -----------------------
    # 4) test에서 클래스별 1장 선택
    # -----------------------
    picks = pick_test_images(TEST_DIR, class_names)

    print("=== Picked test images ===")
    for k, v in picks.items():
        print(f"{k}: {v}")

    # -----------------------
    # 5) Grad-CAM 준비
    # -----------------------
    gradcam = GradCAM(model, target_layer)

    # -----------------------
    # 6) 클래스별 1장씩 Grad-CAM 생성/저장
    # -----------------------
    os.makedirs(OUT_DIR, exist_ok=True)

    for cname in class_names:
        img_path = picks.get(cname, None)
        if img_path is None:
            print(f"[WARN] No image found for class '{cname}' in {TEST_DIR}")
            continue

        orig_pil = Image.open(img_path).convert("RGB")
        orig_np = np.array(orig_pil)

        input_tensor = preprocess(orig_pil).unsqueeze(0).to(DEVICE)

        cam01, pred_idx, pred_prob = gradcam(input_tensor, target_class_idx=None)
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

        # ✅ 원본 파일명 유지 + 충돌 방지
        base = Path(img_path).stem
        save_name = f"GradCAM_{model_name}_GT_{cname}_PRED_{pred_name}_{pred_prob:.3f}_{base}.png"
        save_path = os.path.join(OUT_DIR, save_name)

        text = f"GT:{cname}  Pred:{pred_name} ({pred_prob:.3f})"
        overlay_and_save(orig_np, cam01, save_path, text=text, alpha=0.40, show=True)

    gradcam.remove()
    print("[DONE] Grad-CAM complete.")


if __name__ == "__main__":
    main()
