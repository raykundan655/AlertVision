# all file inside myenv for run python .\myenv\app.py
# we are unable to use whole dataset issue
# 85,000 images → RAM  unable to load this much data into ram because it req 11.2 gb ram that we don't have
# that's why we putting limit 
# Best version in appV1.py 


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from tensorflow.keras.applications import MobileNetV2


img = cv.imread(r"C:\Users\USER\Downloads\mrleyedataset\Open-Eyes\s0001_01981_0_0_1_0_0_01.png")

# if img is None:
#     print("Image not found")
# else:
#     cv.imshow("testing", img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# print(img.shape)
# it is 84*84 but most of transfer learning model use  224*224

datadictionary=r"C:\Users\USER\Downloads\mrleyedataset"

classes=["Open-Eyes","Close-Eyes"]

# for cat in classes:
#     path=os.path.join(datadictionary,cat)
#     for img in os.listdir(path):
#         img_array=cv.imread(os.path.join(path,img),cv.IMREAD_GRAYSCALE)
#         backtogb=cv.cvtColor(img_array,cv.COLOR_GRAY2RGB)
#         # plt.imshow(img_array,cmap="gray")
#         # plt.show()
#         break
#     break



img_size=224

# new_array=cv.resize(backtogb,(img_size,img_size))
# plt.imshow(new_array,cmap="gray")
# plt.show()


training_data=[]

def create_training_data():
    limit = 1000
    for cat in classes:
        path=os.path.join(datadictionary,cat)
        class_num=classes.index(cat)
        count=0
        for img in os.listdir(path):
            if count>=limit:
                break

            count += 1 

            try:
                img_array=cv.imread(os.path.join(path,img),cv.IMREAD_GRAYSCALE)  #Reduce complexity Less noise  Faster processing
                if img_array is None:
                    continue
                
                # Because pretrained models force this requirement: Input shape = (224, 224, 3)
                backtogb=cv.cvtColor(img_array,cv.COLOR_GRAY2RGB)
                new_array=cv.resize(backtogb,(img_size,img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                 print("Error:", e)


create_training_data()

print(len(training_data))

random.shuffle(training_data)

x=[]
y=[]

for feature,label in training_data:
    x.append(feature)
    y.append(label)

x = np.array(x).reshape(-1, img_size, img_size, 3)# net mirror work on 224*224*3
print(x.shape)

x=x/255.0

print(x.shape)

y=np.array(y)




# TRANSFER LEARNING
# tf.keras.applications
#  It is a collection of pretrained deep learning models

model = MobileNetV2(
    input_shape=(img_size, img_size, 3),
    include_top=False,  #Remove last classification layer (1000 classes )
    weights='imagenet'
)

for layer in model.layers:
    layer.trainable = False  #if we don't this then it will start training mobilnet layer and new layer


# For transfer learning (MobileNet) → we use Functional API
#.output it will store the output of last 4th layer


base_input = model.layers[0].input 
base_output = model.layers[-3].output   #remove last 4 layers


x_model = layers.Flatten()(base_output)   # convert to 1D
x_model = layers.Dense(128, activation='relu')(x_model)
x_model = layers.Dense(1, activation='sigmoid')(x_model)

model = tf.keras.Model(inputs=base_input, outputs=x_model)
# we have to give from where it start and where it will end


model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

print(model.summary())

history = model.fit(
    x, y,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)


# testing
def predict_image(path):
    img = cv.imread(path)
    img = cv.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.reshape(img, (1, img_size, img_size, 3))

    pred = model.predict(img)

    if pred[0][0] > 0.5:
        print("Open Eyes")
    else:
        print("Closed Eyes")



predict_image(r"C:\Users\USER\Downloads\mrleyedataset\Open-Eyes\s0001_01842_0_0_1_0_0_01.png")



model.save("mymodel.h5")









# Problem is that we can't load all image into ram that crash the program 
# so we have to make limit ->other solution is below



# solution
# Nobody loads full dataset into RAM
# Instead:
# Load → train → discard → load next










