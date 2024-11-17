import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define paths
data_dir = r'C:\Users\dipes\OneDrive\Desktop\Animals'

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1
)

# Data generators for training and validation
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

mobile_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential([
    mobile_model,
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_generator, epochs=10, validation_data=validation_generator)
model.save('animal_detection.h5')

test_loss, test_acc = model.evaluate(validation_generator)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc * 100)

def predict_and_display_image(model, image_path, threshold=0.95):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  

    predictions = model.predict(img_array)
    max_prob = np.max(predictions[0])  
    class_idx = np.argmax(predictions[0])  
    class_label = list(train_generator.class_indices.keys())[class_idx]

    if max_prob < threshold:
        predicted_label = "No Animal Detected"
    else:
        predicted_label = class_label


    plt.imshow(img)
    plt.title(f'Predicted Label: {predicted_label} (Confidence: {max_prob:.2f})')
    plt.axis('off')
    plt.show()
    return predicted_label

# Test the prediction function
image_path = r'C:\Users\dipes\OneDrive\Desktop\captured_images\animal_20241017_101555.jpg'
predicted_label = predict_and_display_image(model, image_path)
print('Predicted animal:', predicted_label)
