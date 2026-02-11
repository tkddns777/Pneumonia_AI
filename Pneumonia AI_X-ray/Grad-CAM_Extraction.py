import os
import glob
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt
from pathlib import Path
import random

# =====================================================
# 설정: 너 환경에 맞게 여기만 수정
# =====================================================
MODEL_PATH = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia models\resnet18_seed0_epoch002_acc0.960.pth"
TEST_DIR   = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia CT images\test"   # test/NORMAL, test/PNEUMONIA
OUT_DIR    = r"C:\Users\user\OneDrive\바탕 화면\코딩 연습\Pneumonia AI\Grad-CAM\gradcam_test2"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Image_SIZE = 224  # 모델 학습 시 사용한 이미지 크기와 동일하게 설정 필요

# =====================================================
# Unicode-safe 저장 (한글/OneDrive/공백 경로에서도 OK)
# =====================================================
def imwrite_unicode(path, img_bgr):
    """
    Windows에서 cv2.imwrite가 한글/OneDrive 경로에서 실패하는 문제 우회.
    cv2.imencode -> buf.tofile 사용
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext == "":
        ext = ".png"
        path = path.with_suffix(ext)

    # 이미지 dtype/메모리 보정
    img_bgr = np.ascontiguousarray(img_bgr)
    if img_bgr.dtype != np.uint8:
        img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for ext={ext}")

    buf.tofile(str(path))  # 유니코드 경로 OK

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
        self.activations = out  # (N,C,H,W)

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]  # (N,C,H,W)

    def remove(self):
        self.fh.remove()
        self.bh.remove()

    def __call__(self, input_tensor, target_class_idx=None):
        """
        input_tensor: (1,3,H,W)
        target_class_idx: None이면 예측 클래스 사용
        return: cam (H,W) in [0,1], pred_idx, pred_prob
        """
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
    """
    orig_rgb_np: (H,W,3) RGB uint8
    cam_01: (H,W) float in [0,1]
    """
    if orig_rgb_np is None or orig_rgb_np.size == 0:
        raise ValueError("orig_rgb_np is empty.")
    if cam_01 is None or cam_01.size == 0:
        raise ValueError("cam_01 is empty.")

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

    # figure 출력 (NORMAL/PNEUMONIA 각각 1번씩 뜸)
    if show:
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(7, 7))
        plt.imshow(overlay_rgb)
        plt.axis("off")
        plt.title(text if text else "Grad-CAM Overlay")
        plt.tight_layout()
        plt.show()

    # 저장 (cv2.imwrite 사용 금지)
    saved_path = imwrite_unicode(save_path, overlay_bgr)
    print(f"[OK] Saved (unicode-safe): {saved_path}")
    return saved_path


def main():
    # -----------------------
    # 1) 모델 로드
    # -----------------------
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"]

    class_names = checkpoint.get("class_names", ["NORMAL", "PNEUMONIA"])
    num_classes = len(class_names)

    model = models.resnet18(weights=None)

    # fc 구조 맞추기
    if any(k.startswith("fc.1.") for k in state_dict.keys()):
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif any(k.startswith("fc.") for k in state_dict.keys()):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise RuntimeError("state_dict에 fc 키가 없습니다. 저장된 모델 구조를 확인하세요.")

    model.to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # -----------------------
    # 2) 전처리 (학습과 동일해야 가장 정확)
    # -----------------------
    preprocess = transforms.Compose([
        transforms.Resize((Image_SIZE, Image_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # -----------------------
    # 3) test에서 클래스별 1장 선택
    # -----------------------
    picks = pick_test_images(TEST_DIR, class_names)

    print("=== Picked test images ===")
    for k, v in picks.items():
        print(f"{k}: {v}")

    # -----------------------
    # 4) Grad-CAM 준비
    # -----------------------
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    # -----------------------
    # 5) 클래스별 1장씩 Grad-CAM 생성/저장 (파일명 짧게!)
    # -----------------------
    os.makedirs(OUT_DIR, exist_ok=True)

    for idx, cname in enumerate(class_names):
        img_path = picks.get(cname, None)
        if img_path is None:
            print(f"[WARN] No image found for class '{cname}' in {TEST_DIR}")
            continue

        orig_pil = Image.open(img_path).convert("RGB")
        orig_np = np.array(orig_pil)

        input_tensor = preprocess(orig_pil).unsqueeze(0).to(DEVICE)

        cam01, pred_idx, pred_prob = gradcam(input_tensor, target_class_idx=None)
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

        # ✅ 파일명 짧게 (경로 길이 문제 회피)
        # 예: 0_NORMAL_GT_NORMAL_PRED_NORMAL.png
        orig_filename = f"Grad_CAM_{os.path.basename(img_path)}"
        save_path = os.path.join(OUT_DIR, orig_filename)


        text = f"GT:{cname}  Pred:{pred_name} ({pred_prob:.3f})"
        overlay_and_save(orig_np, cam01, save_path, text=text, alpha=0.40, show=True)

    gradcam.remove()
    print("[DONE] Grad-CAM for 2 samples complete.")

    # 저장 폴더 자동 열기(원하면 주석 해제)
    # os.startfile(OUT_DIR)


if __name__ == "__main__":
    main()
