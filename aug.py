import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


base_dir = r'C:\Users\dipes\OneDrive\Desktop\Animals'
folders = ['pig', 'cow', 'ox']

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

for folder in folders:
    img_dir = os.path.join(base_dir, folder)
    save_dir = os.path.join(base_dir, folder)
    os.makedirs(save_dir, exist_ok=True)

    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(img_dir, filename)
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix=folder, save_format='jpeg'):
                i += 1
                if i > 5:
                    break