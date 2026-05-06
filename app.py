import cv2
import torch
import torch.nn as nn
import numpy as np
import math
import json
import time
from torchvision import models, transforms
from PIL import Image
from collections import deque

# ─── Load Config ───────────────────────────────────────────
with open('config.json', 'r') as f:
    config = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─── Model Definition ──────────────────────────────────────
# Confirmed .pth key structure:
#   backbone.features.*         → self.backbone (EfficientNet)
#   sam.*                       → self.sam
#   projector.*                 → self.projector
#   pos_encoder.*               → self.pos_encoder
#   transformer_encoder.*       → self.transformer_encoder
#   classifier.0/2.*            → self.classifier

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class StressNetV3(nn.Module):
    def __init__(self, num_classes=3, seq_length=2, embed_dim=512,
                 num_heads=8, num_layers=2, backbone_name='efficientnet_b0'):
        super().__init__()

        FEATURE_DIMS = {
            'efficientnet_b0': 1280,
            'efficientnet_b1': 1280,
            'efficientnet_b2': 1408,
            'efficientnet_b3': 1536,
            'efficientnet_b4': 1792,
            'efficientnet_b5': 2048,
            'efficientnet_b6': 2304,
            'efficientnet_b7': 2560,
        }

        # ✅ MUST be self.backbone — confirmed from check_keys.py output
        backbone_fn = getattr(models, backbone_name)
        self.backbone = backbone_fn(weights=None)
        feature_dim = FEATURE_DIMS.get(backbone_name, 1280)

        self.sam         = SpatialAttention(kernel_size=7)
        self.avgpool     = nn.AdaptiveAvgPool2d((1, 1))
        self.projector   = nn.Linear(feature_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=0.3, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # ✅ classifier.0 and classifier.2 — matches .pth exactly (Linear, Dropout, Linear)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),   # classifier.0
            nn.Dropout(0.4),             # classifier.1 (no weights, not in .pth)
            nn.Linear(256, num_classes)  # classifier.2
        )

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        features = self.backbone.features(x)   # use .features to skip classifier head
        features = self.sam(features)
        features = self.avgpool(features)
        features = features.flatten(1)
        features = features.view(b, t, -1)
        features = self.projector(features)
        features = features.permute(1, 0, 2)
        features = self.pos_encoder(features)
        features = features.permute(1, 0, 2)
        transformer_out = self.transformer_encoder(features)
        return self.classifier(torch.mean(transformer_out, dim=1))


# ─── Load Model ────────────────────────────────────────────
print("Loading StressNet-V3 model...")
backbone_name = config.get('backbone_name', 'efficientnet_b0')

model = StressNetV3(
    num_classes=config['num_classes'],
    seq_length=config['seq_length'],
    num_heads=config['num_heads'],
    num_layers=config['num_layers'],
    backbone_name=backbone_name
)

state_dict = torch.load('stressnet_v3_model.pth', map_location=device)

# ✅ strict=False only to skip backbone.classifier.1.weight/bias
# (the original EfficientNet head — not used in our model)
missing, unexpected = model.load_state_dict(state_dict, strict=False)

# Verify no important keys are missing
critical_missing = [k for k in missing if not k.startswith('backbone.classifier')]
if critical_missing:
    print(f"[WARN] Missing important keys: {critical_missing}")
else:
    print(f"[OK]   All {len(state_dict)} weights loaded correctly!")
    if unexpected:
        print(f"[INFO] Skipped {len(unexpected)} unused keys: {unexpected}")

model.to(device)
model.eval()
print(f"Model ready! Backbone: {backbone_name}\n")

# ─── Transforms ────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((config['img_size'], config['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─── Settings ──────────────────────────────────────────────
class_names  = ["Low Fatigue", "Moderate Fatigue", "High Fatigue"]
colors       = [(0, 200, 0), (0, 165, 255), (0, 0, 255)]
seq_length   = config['seq_length']
frame_buffer = deque(maxlen=seq_length)

# Per-class thresholds — High Fatigue set lower so it triggers more easily
# Tune these in config.json if needed
CLASS_THRESHOLDS = config.get('class_thresholds', [0.60, 0.60, 0.35])
SMOOTH_WINDOW    = config.get('smooth_window', 5)

prediction_history = deque(maxlen=SMOOTH_WINDOW)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ─── Webcam ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam!")
    exit()

print("Webcam started! Press Q to quit.")
print(f"Thresholds : Low={CLASS_THRESHOLDS[0]:.0%} | "
      f"Moderate={CLASS_THRESHOLDS[1]:.0%} | High={CLASS_THRESHOLDS[2]:.0%}")
print(f"Smoothing  : {SMOOTH_WINDOW} frames\n")

label         = "Warming up..."
confidence    = 0.0
color         = (180, 180, 180)
last_pred_idx = -1
probs_display = None


def draw_prob_bars(frame, probs, x=20, y_start=210):
    bar_colors = [(0, 200, 0), (0, 165, 255), (0, 0, 255)]
    bar_labels = ["Low", "Moderate", "High"]
    bar_w = 150
    for i, (p, bc, lbl) in enumerate(zip(probs, bar_colors, bar_labels)):
        y = y_start + i * 28
        filled = int(bar_w * p)
        cv2.rectangle(frame, (x, y), (x + bar_w, y + 18), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + filled, y + 18), bc, -1)
        cv2.putText(frame, f"{lbl}: {p * 100:.1f}%",
                    (x + bar_w + 8, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)


while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame!")
        break

    display = frame.copy()
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Face Detection Gate ─────────────────────────────────
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    face_detected = len(faces) > 0

    if face_detected:
        x_f, y_f, w_f, h_f = faces[0]
        cv2.rectangle(display, (x_f, y_f), (x_f + w_f, y_f + h_f), (100, 255, 100), 2)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(transform(Image.fromarray(img_rgb)))
    else:
        frame_buffer.clear()
        prediction_history.clear()
        probs_display = None

    # ── Predict when buffer is full ─────────────────────────
    if face_detected and len(frame_buffer) == seq_length:
        seq = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)

        with torch.no_grad():
            probs_np = torch.softmax(model(seq), dim=1)[0].cpu().numpy()

        probs_display = probs_np

        # Priority: check High first → Moderate → Low
        accepted_pred = None
        for class_idx in [2, 1, 0]:
            if probs_np[class_idx] >= CLASS_THRESHOLDS[class_idx]:
                accepted_pred = class_idx
                break

        if accepted_pred is not None:
            prediction_history.append(accepted_pred)

        if len(prediction_history) > 0:
            counts        = np.bincount(list(prediction_history), minlength=3)
            smoothed_pred = int(np.argmax(counts))
            label         = class_names[smoothed_pred]
            confidence    = float(probs_np[smoothed_pred]) * 100
            color         = colors[smoothed_pred]
            last_pred_idx = smoothed_pred
        else:
            label      = "Analyzing..."
            confidence = float(np.max(probs_np)) * 100
            color      = (180, 180, 180)

    elif not face_detected:
        label         = "No Face Detected"
        confidence    = 0.0
        color         = (180, 180, 180)
        last_pred_idx = -1

    # ── HUD Overlay ─────────────────────────────────────────
    overlay = display.copy()
    cv2.rectangle(overlay, (10, 10), (420, 195), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

    cv2.putText(display, "StressNet-V3  |  Fatigue Detection", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.line(display, (20, 48), (410, 48), (60, 60, 60), 1)
    cv2.putText(display, f"State:  {label}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    if confidence > 0:
        cv2.putText(display, f"Conf:   {confidence:.1f}%", (20, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)
    face_col = (0, 220, 0) if face_detected else (80, 80, 255)
    cv2.putText(display,
                "Face: DETECTED" if face_detected else "Face: NOT FOUND",
                (20, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_col, 1)
    cv2.putText(display, time.strftime('%H:%M:%S'), (20, 172),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

    # Alert banners
    h_f2, w_f2 = display.shape[:2]
    if last_pred_idx == 1:
        cv2.rectangle(display, (0, h_f2 - 40), (w_f2, h_f2), (0, 100, 180), -1)
        cv2.putText(display, "  TAKE A SHORT BREAK!",
                    (10, h_f2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    elif last_pred_idx == 2:
        cv2.rectangle(display, (0, h_f2 - 40), (w_f2, h_f2), (0, 0, 200), -1)
        cv2.putText(display, "  MANDATORY BREAK NOW!",
                    (10, h_f2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    if probs_display is not None:
        draw_prob_bars(display, probs_display, x=20, y_start=210)

    cv2.imshow("StressNet-V3 | Fatigue Detection", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("App closed.")