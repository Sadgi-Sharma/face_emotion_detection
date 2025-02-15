# Import required libraries for data processing, machine learning, and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow
from tensorflow.keras.utils import to_categorical
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import cv2
import warnings

# Load the Facial Emotion Recognition (FER) 2013 dataset
data = pd.read_csv("fer2013.csv")
# Basic dataset exploration
data.shape  # Check the dimensions of the dataset
data.head(3)  # View the first 3 rows
data.isnull().sum()  # Check for null values
data["Usage"].value_counts()  # Count of data usage categories
data["emotion"].value_counts()  # Count of different emotions


# Function to convert pixel string to image array
def show_img(df_row, pixel_string_col, class_col):
    # Extract pixels and label from dataframe row
    pixels = df_row[pixel_string_col]
    label = df_row[class_col]
    # Reshape pixels into 48x48 matrix
    pic = np.array(pixels.split())
    pic = pic.reshape(48, 48)
    # Create a 3-channel image (grayscale duplicated across channels)
    image = np.zeros((48, 48, 3))
    image[:, :, 0] = pic
    image[:, :, 1] = pic
    image[:, :, 2] = pic
    return np.array([image.astype(np.uint8), label])


# Visualize images for each emotion
import matplotlib.pyplot as plt

for emotion in range(1, 8):
    # Select first image for each emotion
    picture = data[data["emotion"] == emotion - 1].iloc[0]
    picture = show_img(picture, "pixels", "emotion")
    # Display image with corresponding emotion
    plt.imshow(picture[0])
    plt.title(picture[1])
    plt.show()
# Create a dictionary to map emotion codes to readable labels
emo_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}
# Create a copy of the dataset with readable emotion labels
df1 = data.copy()
df1["emotion"] = df1["emotion"].map(emo_dict)
# Visualize emotion distribution
sns.countplot(x=df1["emotion"])
plt.title("Emotion Distribution")
# Split data into training, validation, and test sets
train_data = data[data["Usage"] == "Training"]
val_data = data[data["Usage"] == "PublicTest"]
test_data = data[data["Usage"] == "PrivateTest"]
# Print shape of each dataset
print(
    "The shape of training set is: {}, \nThe shape of validation set is: {}, \nThe shape of test set is: {}".format(
        train_data.shape, val_data.shape, test_data.shape
    )
)
# Reset index and remove 'Usage' column for all datasets
for elem in [train_data, val_data, test_data]:
    elem.reset_index(drop=True, inplace=True)
    elem.drop("Usage", axis=1, inplace=True)

# Preprocess test data
test_data["pixels"] = test_data["pixels"].apply(
    lambda pixels: [int(pixel) for pixel in pixels.split()]
)
X_test = (
    np.array(test_data["pixels"].tolist(), dtype="float32").reshape(-1, 48, 48, 1)
    / 255.0
)
y_test = to_categorical(test_data["emotion"], num_classes=7, dtype="uint8")
# Define base path for saving images
path = "./"
# Function to create directory structure for training and validation data
def create_dir(path, class_list):
    train_path = os.path.join(path, "train")
    val_path = os.path.join(path, "valid")
    # Create train and validation directories
    os.mkdir(train_path)
    os.mkdir(val_path)
    # Create subdirectories for each emotion class
    for data_path, cat in {train_path: "train-", val_path: "valid-"}.items():
        for label in class_list:
            label_dir = os.path.join(data_path, cat + str(label))
            os.mkdir(label_dir)


# Create directories for emotion classes
create_dir(path, [0, 1, 2, 3, 4, 5, 6])
# Define paths for training and validation image directories
train_path = "./train"
val_path = "./valid"
# Function to save images to respective emotion class directories
def save_imgs(df, df_path, pixel_col, class_col, class_list, prefix):
    for i in range(len(df)):
        pixel_string = df[pixel_col][i]
        pixels = list(map(int, pixel_string.split()))
        # Convert pixel data to image
        matrix = np.array(pixels).reshape(48, 48).astype(np.uint8)
        img = Image.fromarray(matrix)

        for label in class_list:
            if str(df[class_col][i]) in prefix + str(label):
                img.save(
                    df_path
                    + "/"
                    + prefix
                    + str(label)
                    + "/"
                    + prefix
                    + str(label)
                    + "-"
                    + str(i)
                    + ".png"
                )
            else:
                continue


# Save training and validation images
save_imgs(train_data, train_path, "pixels", "emotion", [0, 1, 2, 3, 4, 5, 6], "train-")
save_imgs(val_data, val_path, "pixels", "emotion", [0, 1, 2, 3, 4, 5, 6], "valid-")
# Import image data generator for data augmentation
from keras.preprocessing.image import ImageDataGenerator

# Function to create data generators with augmentation
def img_data_gen(train_path, val_path, target_size, batch_size, color_mode, class_mode):
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode="nearest",
    )
    # Validation data generator (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    # Create generators for training and validation data
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=target_size,
        batch_size=batch_size,
        color_mode=color_mode,
        seed=42,
        shuffle=True,
        class_mode=class_mode,
    )
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=target_size,
        batch_size=batch_size,
        color_mode=color_mode,
        seed=42,
        shuffle=True,
        class_mode=class_mode,
    )
    return train_generator, val_generator


# Create data generators
train_gen, val_gen = img_data_gen(
    train_path,
    val_path,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
)
# Create Sequential Convolutional Neural Network model
model = Sequential()
num_classes = 7
# First convolutional block
model.add(
    Convolution2D(
        64, kernel_size=3, activation="relu", padding="same", input_shape=(48, 48, 1)
    )
)
model.add(BatchNormalization())
model.add(Convolution2D(64, kernel_size=3, activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Second convolutional block
model.add(Convolution2D(128, kernel_size=3, activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(Convolution2D(128, kernel_size=3, activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(Convolution2D(128, kernel_size=3, activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
# Third convolutional block
model.add(Convolution2D(256, kernel_size=3, activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(Convolution2D(256, kernel_size=3, activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(Convolution2D(256, kernel_size=3, activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
# Fourth convolutional block
model.add(Convolution2D(256, kernel_size=3, activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(Convolution2D(256, kernel_size=3, activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(Convolution2D(256, kernel_size=3, activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
# Flatten and output layer
model.add(Flatten())
model.add(Dense(num_classes, activation="softmax"))
# Display model summary
model.summary()
# Configure optimizer and compile the model
optimizer = Adam(lr=0.0001, decay=1e-6)
model.compile(
    loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)
# Early stopping callback to prevent overfitting
earlystop = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=15, verbose=1, restore_best_weights=True
)
# Train the model
epochs = 100
history = model.fit_generator(
    train_gen,
    steps_per_epoch=len(train_data) // 64,
    epochs=epochs,
    callbacks=earlystop,
    verbose=1,
    validation_data=val_gen,
    validation_steps=len(val_data) // 64,
)
# Convert training history to DataFrame for visualization
model_df = pd.DataFrame(model.history.history)
# Plot training and validation loss
model_df[["loss", "val_loss"]].plot()
plt.title("Loss Plots for Training and Validation Sets")
# Plot training and validation accuracy
model_df[["accuracy", "val_accuracy"]].plot()
plt.title("Accuracy Plots for Training and Validation Sets")

# Evaluate model on test data
true_y = np.argmax(y_test, axis=1)
pred_y = np.argmax(model.predict(X_test), axis=1)
print(classification_report(true_y, pred_y))
# Save the trained model
model.save("emotion_detection.h5")
