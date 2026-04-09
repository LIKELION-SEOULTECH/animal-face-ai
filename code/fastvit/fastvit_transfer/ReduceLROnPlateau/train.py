import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.models import create_model
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import random
import numpy as np

# 재현성을 위한 시드 고정
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    seed_everything(42) # 시드 고정으로 실행 시마다 결과 일관성 유지

    # --- 1. 경로 설정 ---
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
    
    train_dir = os.path.join(project_root, 'data', 'train')
    val_dir = os.path.join(project_root, 'data', 'val')
    
    print(f"📂 Train 경로: {train_dir}")
    print(f"📂 Val 경로: {val_dir}")

    # --- 2. 하이퍼파라미터 설정 ---s
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_epochs = 100
    learning_rate = 1e-4
    input_size = 224 # FastViT 최적 입력 크기

    # --- 3. 데이터 전처리 ---
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

    try:
        image_datasets = {
            'train': datasets.ImageFolder(train_dir, data_transforms['train']),
            'val': datasets.ImageFolder(val_dir, data_transforms['val'])
        }
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=0) for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        print(f"✅ 데이터 로드 완료! 클래스: {class_names}")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return

    # --- 4. 모델 설정 ---
    model = create_model("fastvit_t8", pretrained=True, num_classes=len(class_names))
    model = model.to(device)

    # --- 5. 손실함수 및 최적화 설정 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # --- 6. 조기 종료 설정 ---
    early_stop_patience = 15  
    early_stop_counter = 0
    best_loss = float('inf')
    best_acc = 0.0
    early_stop_flag = False

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # --- 7. 학습 루프 ---
    print(f"🔥 학습 시작 (Early Stopping Patience: {early_stop_patience})")

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

            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}/{num_epochs-1} [{phase}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} LR: {current_lr:.6f}')

            if phase == 'val':
                scheduler.step(epoch_loss)

                # 최적 모델 저장
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 'fastvit_ReduceLROnPlateau.pth')
                    print(f"🌟 Best Accuracy 갱신: {best_acc:.4f}")

                # Early Stopping 로직
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    early_stop_counter = 0 
                else:
                    early_stop_counter += 1
                    print(f"📉 ES 카운트: {early_stop_counter}/{early_stop_patience}")
                    if early_stop_counter >= early_stop_patience:
                        print("🛑 조기 종료 조건 충족!")
                        early_stop_flag = True

    # --- 8. 결과 시각화 ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val'); plt.title('Loss'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history['train_acc'], label='Train'); plt.plot(history['val_acc'], label='Val'); plt.title('Acc'); plt.legend()
    plt.tight_layout()
    plt.savefig('fastvit_ReduceLROnPlateau_result.png')
    print("🎨 결과 그래프 저장 완료.")

if __name__ == '__main__':
    main()