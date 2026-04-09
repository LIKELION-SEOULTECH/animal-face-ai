import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 경로 및 파라미터 설정
train_dir = '../../../../data/train'
test_dir = '../../../../data/val'
model_save_path = 'cnn_data_aug.h5'

img_width, img_height = 256, 256
batch_size = 16
epochs = 100 # early stopping 적용하므로

# 2. 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,      # 최대 30도 회전
    width_shift_range=0.2,   # 가로 이동
    height_shift_range=0.2,  # 세로 이동
    shear_range=0.2,         # 전단 변환 (기울기)
    zoom_range=0.2,          # 확대/축소
    horizontal_flip=True,    # 좌우 반전
    fill_mode='nearest'      # 빈 공간 채우기
)

# 검증 데이터는 변형 없이 진행
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical'
)
validation_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical'
)

# 3. 모델 구성
model = Sequential([
    Input(shape=(img_width, img_height, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    # 드롭아웃 비율을 유지하여 과적합 방지
    Dropout(0.5), 
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# 4. 컴파일 (라벨 스무딩으로 확신 과잉 방지)
model.compile(
    optimizer=Adam(learning_rate=1e-4), 
    loss=CategoricalCrossentropy(label_smoothing=0.1), 
    metrics=['accuracy']
)

# 5.  patience=15: val_loss가 15번의 에폭 동안 개선되지 않으면 멈춤
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True, 
    verbose=1
)

checkpoint = ModelCheckpoint(
    model_save_path, 
    monitor='val_loss', 
    save_best_only=True, 
    verbose=1
)

# 6. 학습 시작
print("🔥 고강도 증강 모드로 학습을 시작합니다...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint]
)

# 7. 결과 저장 및 시각화
def save_final_plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title('Final Model Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Final Model Loss')
    plt.legend()

    plt.savefig('cnn_data_aug_result.png')
    print("📊 학습 결과가 'cnn_data_aug_result.png'로 저장되었습니다.")

save_final_plot(history)