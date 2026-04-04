# 85,000 images → load into RAM → crash
# Load small batch → train → remove → load next batch 
# This is exactly what ImageDataGenerator does.
# ImageDataGenerator->It is a data loader + preprocessor + batch maker

import tensorflow as tf
from tensorflow.keras  import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import numpy as np

dictt=r"C:\Users\USER\Downloads\mrleyedataset"
img_size=224
batch_size=32

# data generator

datagen=ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2
)
# It does NOT load images,It just sets rules for how images will be processed later
# It does NOT split files physically  It just splits internally when you use:
# subset='training'
# subset='validation'

# flow_from_directory():"Automatically read images from folders, label them, and feed them to the model in small batches"

train_data=datagen.flow_from_directory(
dictt,
target_size=(img_size,img_size),
batch_size=batch_size,
class_mode="binary" ,  #assign 0 and 1 to the inserted folder 0->open eye 1->close eye
subset="training"
)


valid_data=datagen.flow_from_directory(
    dictt,
    target_size=(img_size,img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)

print("Class indices:", train_data.class_indices)  #it will show which class have which indices


model=MobileNetV2(
    input_shape=(img_size,img_size,3),
    include_top=False,
    weights="imagenet"
)


for layer in model.layers:
    layer.trainable=False

x=model.output
x=layers.Flatten()(x)
x=layers.Dense(128,activation="relu")(x)
x=layers.Dense(1,activation="sigmoid")(x)

model=tf.keras.Model(inputs=model.input,outputs=x)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# print(model.summary())

history=model.fit(
    train_data,
    validation_data=valid_data, #after testing model silently validate
    epochs=1
)

model.save("mymodel.h5")
print("model is saved")


def testing(path):
    img=cv.imread(path)
    if img is None:
        print("image not found")
        return

    img=cv.resize(img,(img_size,img_size))
    img=img/255.0
    img = img.reshape(1, img_size, img_size, 3)

    pred=model.predict(img)

    if(pred[0][0]>0.5):
        print("open eyes")
    else:
        print("close eye")





testing(r"C:\Users\USER\Downloads\mrleyedataset\Close-Eyes\s0001_00002_0_0_0_0_0_01.png")
