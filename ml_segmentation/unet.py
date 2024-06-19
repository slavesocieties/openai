import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate

def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)

    # Decoder
    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop4))
    merge6 = concatenate([conv3, up6], axis=3)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv2, up7], axis=3)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv1, up8], axis=3)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # Output layer
    conv9 = Conv2D(num_classes, 1, activation='softmax')(conv8)

    model = Model(inputs=inputs, outputs=conv9)

    return model

# Define input shape and number of classes
input_shape = (256, 192, 1)
num_classes = 2  # e.g., "text", "folio", "background", "ruler", "finger"

# Build the U-Net model
model = build_unet(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load your training and validation data
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

def mask_to_onehot(mask, num_classes):
    """
    Converts a grayscale segmentation mask to a one-hot encoded mask.
    
    Args:
        mask (numpy.ndarray): The grayscale segmentation mask, where pixel values represent class labels.
        num_classes (int): The number of classes in the segmentation task.
        
    Returns:
        numpy.ndarray: The one-hot encoded segmentation mask.
    """
    one_hot_mask = np.zeros((mask.shape[0], mask.shape[1], num_classes), dtype=np.uint8)
    for c in range(num_classes):
        one_hot_mask[:, :, c] = (mask == c).astype(np.uint8)
    return one_hot_mask

def load_data(image_dir, mask_dir, image_size, num_classes):
    images = []
    masks = []
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    for img_file, mask_file in zip(image_files, mask_files):
        # Load and preprocess image
        img_path = os.path.join(image_dir, img_file)
        img = load_img(img_path, color_mode='grayscale', target_size=(image_size[1], image_size[0]))  # (width, height)
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
        images.append(img)
        
        # Load and preprocess mask
        mask_path = os.path.join(mask_dir, mask_file)
        mask = load_img(mask_path, color_mode='grayscale', target_size=(image_size[1], image_size[0]))  # (width, height)
        mask = img_to_array(mask)
        mask = mask[:, :, 0]  # Remove the extra channel dimension
        #mask = to_categorical(mask, num_classes=num_classes)  # One-hot encode the mask
        mask = mask_to_onehot(mask, 2)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Define your image and mask directories
image_dir = 'images/resized'
mask_dir = 'masks'
image_size = (192, 256)  # (width, height)
num_classes = 2  # e.g., "text", "folio", "background", "ruler", "finger"

# Load data
images, masks = load_data(image_dir, mask_dir, image_size, num_classes)

# Perform a randomized train/validation split
train_images, val_images, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42)

print(f"Training images: {train_images.shape}")
print(f"Validation images: {val_images.shape}")
print(f"Training masks: {train_masks.shape}")
print(f"Validation masks: {val_masks.shape}")

# Example usage with your U-Net model
model.fit(train_images, train_masks, validation_data=(val_images, val_masks), epochs=20, batch_size=8)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(val_images, val_masks)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

import matplotlib.pyplot as plt

def preprocess_new_image(image_path, image_size):
    # Load and preprocess the new image
    img = load_img(image_path, color_mode='grayscale', target_size=(image_size[1], image_size[0]))  # (width, height)
    img = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def postprocess_mask(predicted_mask):
    # Convert predicted mask from one-hot encoding to class labels
    predicted_mask = np.argmax(predicted_mask, axis=-1)
    predicted_mask = np.squeeze(predicted_mask)  # Remove batch dimension
    return predicted_mask

# Load and preprocess a new image
new_image_path = '/path/to/your/new_image.jpg'
new_image = preprocess_new_image(new_image_path, image_size)

# Predict the segmentation mask for the new image
predicted_mask = model.predict(new_image)
predicted_mask = postprocess_mask(predicted_mask)

# Display the new image and the predicted mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('New Image')
plt.imshow(np.squeeze(new_image), cmap='gray')  # Remove batch dimension and display
plt.subplot(1, 2, 2)
plt.title('Predicted Mask')
plt.imshow(predicted_mask, cmap='gray')
plt.show()
