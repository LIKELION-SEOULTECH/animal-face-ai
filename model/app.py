import onnxruntime as ort
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
import uvicorn
import io

app = FastAPI()

# 1. ONNX 모델 로드
ONNX_PATH = "fastvit.onnx"
session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

def preprocess(image_bytes):
    # 바이트 데이터를 numpy 배열로 변환
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 전처리 로직
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.transpose(2, 0, 1).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    
    img /= 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    img = (img - mean) / std
    return img.astype(np.float32)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. 이미지 읽기
    image_bytes = await file.read()
    input_data = preprocess(image_bytes)
    
    # 2. 모델 추론
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)
    
    # 3. 결과 해석
    result = outputs[0]
    predicted_idx = int(np.argmax(result))
    confidence = float(np.max(torch_softmax(result))) 
    
    return {
        "prediction": predicted_idx,
        "class_name": f"Class {predicted_idx}", 
        "status": "success"
    }

@app.get("/")
def health_check():
    return {"status": "ok", "message": "FastViT ONNX Server is running"}

def torch_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)