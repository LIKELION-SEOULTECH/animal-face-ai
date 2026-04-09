import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. 경로 설정 (딕셔너리로 안전하게 관리) ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_paths = {
        'train': os.path.normpath(os.path.join(current_dir, "..", "..", "..", "data", "train")),
        'val': os.path.normpath(os.path.join(current_dir, "..", "..", "..", "data", "val"))
    }

    # 경로 존재 확인
    for phase, path in data_paths.items():
        if not os.path.exists(path):
            print(f"❌ 경로를 찾을 수 없습니다 ({phase}): {path}")
            return

    # --- 2. 하이퍼파라미터 ---
    batch_size = 32
    num_epochs = 60
    learning_rate = 1e-4
    input_size = 224

    # --- 3. 데이터 증강 ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 에러가 발생했던 ImageFolder 로드 부분 수정
    image_datasets = {
        x: datasets.ImageFolder(data_paths[x], data_transforms[x]) 
        for x in ['train', 'val']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=0) 
        for x in ['train', 'val']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"✅ 데이터 로드 완료! 클래스: {class_names} | 총 {sum(dataset_sizes.values())}장")

    # --- 4. 모델 설정 (ResNet18) ---
    print("🚀 ResNet18 모델 로드 중...")
    model = models.resnet18(weights='IMAGENET1K_V1') # 최신 버전의 가중치 로드 방식
    num_ftrs = model.fc.in_features
    
    # 마지막 레이어 교체 (Dropout 추가로 오버피팅 방지)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(class_names))
    )
    model = model.to(device)

    # --- 5. 손실함수 및 옵티마이저 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    # 20에폭마다 학습률을 1/10로 감소
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # --- 6. 학습 루프 ---
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"🔥 학습 시작 (Device: {device})")
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

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

            if phase == 'val':
                scheduler.step()
                print(f'Epoch {epoch}/{num_epochs-1} | Train Acc: {history["train_acc"][-1]:.4f} | Val Acc: {epoch_acc:.4f}')
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 'resnet18_animal_best.h5')
                    print(f"🌟 Best Accuracy 갱신: {best_acc:.4f} (저장됨)")

    # --- 7. 결과 그래프 저장 ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val')
    plt.title('ResNet18 Loss'); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train'); plt.plot(history['val_acc'], label='Val')
    plt.title('ResNet18 Accuracy'); plt.legend()
    
    plt.tight_layout()
    plt.savefig('resnet18_training_result.png')
    print("🎨 결과 분석 그래프가 'resnet18_training_result.png'로 저장되었습니다.")

if __name__ == '__main__':
    main()