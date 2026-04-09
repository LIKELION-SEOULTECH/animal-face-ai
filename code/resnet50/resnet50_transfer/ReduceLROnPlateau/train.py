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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # 추가

# 1. 파라미터 설정
img_width, img_height = 256, 256
batch_size = 16
epochs = 100
model_save_path = 'resnet50_ReduceLROnPlateau.h5' 

# 2. 데이터 생성기 
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
x = GlobalAveragePooling2D()(x) 
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

# 5. 콜백 설정 (ReduceLROnPlateau 추가)
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

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.1,          # 학습률을 줄일 비율 (LR * 0.1)
    patience=5,           # 참을 에폭 수
    min_lr=1e-7,         # 최소 학습률 한계
    verbose=1            # 학습률 변경 시 메시지 출력
)

# 6. 학습 시작
print("🔥 ResNet50 전이 학습 시작 (with ReduceLROnPlateau)...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint, reduce_lr] # 콜백 리스트에 추가
)

# 7. 결과 저장
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
    plt.savefig('resnet50_ReduceLROnPlateau_result.png')
    print("🎨 결과 그래프가 'resnet50_ReduceLROnPlateau_result.png'로 저장되었습니다.")

save_resnet_plot(history)