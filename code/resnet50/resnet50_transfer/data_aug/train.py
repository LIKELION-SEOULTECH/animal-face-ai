import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 파라미터 설정
img_width, img_height = 256, 256
batch_size = 16
epochs = 100
model_save_path = 'resnet50_data_aug.h5'

# 2. 데이터 생성기 (기존의 강력한 증강 유지)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
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

# 3. ResNet50 모델 구성
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
base_model.trainable = True 

x = base_model.output
x = GlobalAveragePooling2D()(x) # 특징 맵을 벡터화
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. 컴파일
model.compile(
    optimizer=Adam(learning_rate=1e-4), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 콜백 및 학습
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)

print("🔥 ResNet50 전이 학습 시작...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint]
)

# 6. 결과 저장
def save_resnet_plot(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('ResNet50 Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('ResNet50 Loss')
    plt.legend()
    plt.savefig('resnet50_data_aug_result.png')

save_resnet_plot(history)