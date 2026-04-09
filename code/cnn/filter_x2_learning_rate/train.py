import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 파라미터 설정
img_width, img_height = 256, 256
batch_size = 16
epochs = 100 
model_save_path = 'cnn_filter_x2_learning_rate.h5'

# 2. 데이터 생성기
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '../../../../data/train', target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical'
)
validation_generator = test_datagen.flow_from_directory(
    '../../../../data/val', target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical'
)

# 3. 모델 구성 (필터 2배 + 배치 정규화 추가)
model = Sequential([
    Input(shape=(img_width, img_height, 3)),
    
    # Block 1: 64 filters
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Block 2: 128 filters
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Block 3: 256 filters
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Block 4: 512 filters 추가
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dropout(0.5), 
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# 4. 컴파일
model.compile(
    optimizer=Adam(learning_rate=2e-4), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# 5. 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)

# 6. 학습
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint]
)

# 7. 그래프 저장 
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

    plt.savefig('cnn_filter_x2_learning_rate_result.png')
    print("📊 학습 결과가 'cnn_filter_x2_learning_rate_result.png'로 저장되었습니다.")

save_final_plot(history)