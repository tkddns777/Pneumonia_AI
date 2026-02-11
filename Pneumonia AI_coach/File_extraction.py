import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# =====================================================
# 경로 설정
# =====================================================
INPUT_ROOT = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Respiratory_Sound_Database\Respiratory_Sound_Database"
OUTPUT_ROOT = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Respiratory_Sound_Database\Respiratory_Sound_Database\MelSpectrogram_Output"

CLASSES = ["Pneumonia", "Healthy"]

# =====================================================
# Mel-spectrogram 파라미터
# =====================================================
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
DURATION = 5.0  # seconds

# =====================================================
# 출력 폴더 생성 (클래스별)
# =====================================================
for cls in CLASSES:
    os.makedirs(os.path.join(OUTPUT_ROOT, cls), exist_ok=True)

# =====================================================
# wav → mel-spectrogram 저장 함수
# =====================================================
def save_mel(wav_path, save_path):
    try:
        y, sr = librosa.load(
            wav_path,
            sr=SR,
            mono=True,
            duration=DURATION
        )

        # 길이 보정 (padding / truncate)
        target_len = int(SR * DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        plt.figure(figsize=(3, 3))
        librosa.display.specshow(mel_db, sr=sr, hop_length=HOP_LENGTH)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    except Exception as e:
        print(f"[ERROR] {wav_path} → {e}")

# =====================================================
# 클래스별 변환 실행
# =====================================================
for cls in CLASSES:
    input_dir = os.path.join(INPUT_ROOT, cls)
    output_dir = os.path.join(OUTPUT_ROOT, cls)

    wav_files = list(Path(input_dir).glob("*.wav"))

    print(f"\n▶ {cls}: {len(wav_files)} files")

    for wav_path in tqdm(wav_files):
        save_path = os.path.join(
            output_dir,
            wav_path.stem + ".png"
        )
        save_mel(str(wav_path), save_path)

print("\n✅ Pneumonia / Healthy 폴더별 Mel-spectrogram 저장 완료")
