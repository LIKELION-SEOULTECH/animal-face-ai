import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 경로 및 파라미터 설정
train_dir = '../../../../data/train'
test_dir = '../../../../data/val'
model_save_path = 'resnet50_baseline.h5'

img_width, img_height = 256, 256
batch_size = 16
epochs = 100

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

# 3. ResNet50 모델 구성
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. 컴파일
model.compile(
    optimizer=Adam(learning_rate=1e-5), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)

# 6. 학습 시작
print("🚀 ResNet50 베이스라인 학습 시작...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint]
)

# 7. 시각화 및 결과 저장
def save_plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title('ResNet50 Baseline Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('ResNet50 Baseline Loss')
    plt.legend()

    plt.savefig('resnet50_baseline_result.png')
    print("📊 결과가 'resnet50_baseline_result.png'로 저장되었습니다.")

save_plot(history)