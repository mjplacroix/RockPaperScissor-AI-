import numpy as np
import cv2
import os
import sys

# DATA_PATH = sys.argv[1] # Path to folder containing data
PATH = os.getcwd()
# path = sys.argv[1]

DATA_PATH = os.path.join(PATH)

shape_to_label = {'rock':np.array([1.,0.,0.,0.]),'paper':np.array([0.,1.,0.,0.]),'scissor':np.array([0.,0.,1.,0.]),'ok':np.array([0.,0.,0.,1.])}
arr_to_shape = {np.argmax(shape_to_label[x]):x for x in shape_to_label.keys()}

imgData = list()
labels = list()

for dr in os.listdir(DATA_PATH):
    if dr not in ['rock','paper','scissor']:
        continue
    print(dr)
    lb = shape_to_label[dr]
    i = 0
    for pic in os.listdir(os.path.join(DATA_PATH,dr)):
        path = os.path.join(DATA_PATH,dr+'/'+pic)
        img = cv2.imread(path)
        print(img.shape)
        imgData.append([img,lb])
        imgData.append([cv2.flip(img, 1),lb]) #horizontally flipped image
        imgData.append([cv2.resize(img[50:250,50:250],(300,300)),lb]) # zoom : crop in and resize
        i+=3
    print(i)

np.random.shuffle(imgData)

imgData,labels = zip(*imgData)

imgData = np.array(imgData)
labels = np.array(labels)