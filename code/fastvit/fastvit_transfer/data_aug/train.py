import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.models import create_model
import matplotlib
matplotlib.use('Agg') # GUI 에러 방지
import matplotlib.pyplot as plt

def main():
    # --- 1. 상대 경로 설정 ---
    # 현재 파일: animal-face-ai/code/train/fastvit_transfer/baseline/train.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 데이터 경로 (상대 경로로 4단계 위로 이동 후 data 폴더)
    train_dir = os.path.normpath(os.path.join(current_dir, "..", "..", "..", "..", "data", "train"))
    val_dir = os.path.normpath(os.path.join(current_dir, "..", "..", "..", "..", "data", "val"))

    # ml-fastvit 라이브러리 경로 (필요한 경우)
    project_root = os.path.normpath(os.path.join(current_dir, "..", "..", "..", ".."))
    ml_fastvit_path = os.path.join(project_root, 'ml-fastvit')
    if os.path.exists(ml_fastvit_path):
        sys.path.append(ml_fastvit_path)

    print(f"📂 Train 경로: {train_dir}")
    print(f"📂 Val 경로: {val_dir}")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"❌ 데이터를 찾을 수 없습니다: {train_dir}")

    # --- 2. 하이퍼파라미터 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_epochs = 100
    learning_rate = 1e-4
    input_size = 224  # FastViT 표준 사이즈로 변경 (Shape 에러 방지 핵심)

    # --- 3. 데이터 전처리 및 증강 ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, data_transforms['val'])
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=0) 
        for x in ['train', 'val']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"✅ 데이터 로드 완료! 클래스: {class_names}")

    # --- 4. 모델 설정 ---
    # num_classes를 직접 지정하여 모델이 자동으로 적절한 Head를 생성하게 함
    model = create_model("fastvit_t8", pretrained=True, num_classes=len(class_names))
    model = model.to(device)

    # --- 5. 손실함수, 옵티마이저, 스케줄러 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # --- 6. Early Stopping 설정 ---
    early_stop_patience = 15 
    early_stop_counter = 0
    best_loss = float('inf')
    best_acc = 0.0
    early_stop_flag = False

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # --- 7. 학습 루프 ---
    print(f"🔥 학습 시작 (Device: {device})")

    for epoch in range(num_epochs):
        if early_stop_flag: break 

        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'Epoch {epoch}/{num_epochs - 1} [{phase}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                scheduler.step(epoch_loss)

                # Best Accuracy 저장
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 'fastvit_data_aug.pth')
                    print(f"🌟 Best Accuracy 갱신: {best_acc:.4f}")

                # Early Stopping 체크
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    print(f"📉 EarlyStopping 카운트: {early_stop_counter}/{early_stop_patience}")
                    if early_stop_counter >= early_stop_patience:
                        print("🛑 조기 종료!")
                        early_stop_flag = True

    # --- 8. 결과 그래프 저장 ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val')
    plt.title('Loss'); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train'); plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy'); plt.legend()
    
    plt.tight_layout()
    plt.savefig('fastvit_data_aug_result.png')
    print("🎨 결과 그래프 저장 완료.")

if __name__ == '__main__':
    main()