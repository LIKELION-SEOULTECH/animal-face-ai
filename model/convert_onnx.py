import os
import torch
from timm.models import create_model

# ─── 설정 ───────────────────────────────────────────
MODEL_PATH  = "fastvit_data_aug.pth"
ONNX_PATH   = "fastvit.onnx"
NUM_CLASSES = 3
INPUT_SIZE  = (1, 3, 224, 224)
OPSET       = 13
# ────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  사용 장치: {device}")

# 1. 모델 파일 존재 확인
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ 모델 파일이 없습니다! 경로를 확인해주세요.\n탐색 경로: {MODEL_PATH}"
    )

print(f"✅ 모델 파일 확인: {MODEL_PATH}")

# 2. 모델 구조 생성
model = create_model("fastvit_t8", pretrained=False, num_classes=NUM_CLASSES)

# 3. 가중치 로드
state_dict = torch.load(MODEL_PATH, map_location=device)

# state_dict가 dict 안에 감싸진 경우 대응 (예: {"model": {...}} 형태)
if "model" in state_dict:
    state_dict = state_dict["model"]
elif "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("🚀 가중치 로드 완료!")
total_params = sum(p.numel() for p in model.parameters())
print(f"📋 총 파라미터 수: {total_params:,}개")

# 4. ONNX 내보내기
dummy_input = torch.randn(*INPUT_SIZE).to(device)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    opset_version=OPSET,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print(f"✅ ONNX 내보내기 완료: {ONNX_PATH}")