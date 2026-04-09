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

# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 사용 가능: {len(gpus)}개 감지")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ GPU를 찾을 수 없습니다. CPU로 진행합니다.")

# 1. 경로 및 파라미터 설정
train_dir = '../../../../data/train'
test_dir = '../../../../data/val'
model_save_path = 'cnn_baseline_best.h5' # 가장 성능 좋은 시점 저장용

img_width, img_height = 256, 256
batch_size = 16
epochs = 100 # early stopping 적용하므로

# 2. 최소한의 전처리 
train_datagen = ImageDataGenerator(rescale=1./255)
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
    Dropout(0.5), 
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# 4. 컴파일
model.compile(
    optimizer=Adam(learning_rate=1e-4), 
    loss=CategoricalCrossentropy(label_smoothing=0.1), 
    metrics=['accuracy']
)

# 5. patience=15: val_loss가 15번의 에폭 동안 개선되지 않으면 멈춤
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True, 
    verbose=1
)

# 학습 도중 val_loss가 가장 낮은 '최적의 모델'을 자동으로 저장
checkpoint = ModelCheckpoint(
    model_save_path, 
    monitor='val_loss', 
    save_best_only=True, 
    verbose=1
)

# 6. 모델 학습
print("🚀 베이스라인 학습 시작...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint] # 콜백 적용
)

# 7. 시각화 및 결과 저장
def save_plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    plt.title('Baseline Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title('Baseline Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('cnn_baseline_result.png')
    print("📊 결과가 'cnn_baseline_result.png'로 저장되었습니다.")

save_plot(history)