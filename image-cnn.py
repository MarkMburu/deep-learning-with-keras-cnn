# %%
"""

"""

# %%
"""
### Activate virtual environment
"""

# %%
%%bash
source ~/kerai/bin/activate

# %%
"""
### Imports
"""

# %%
%matplotlib inline
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation

# %%
"""
Import helper functions
"""

# %%
from helper import get_class_names, get_train_data, get_test_data, plot_images, plot_model

# %%
"""
Change matplotlib graph style
"""

# %%
matplotlib.style.use('ggplot')

# %%
"""
### Constants
"""

# %%
"""
Import class names
"""

# %%
class_names = get_class_names()
print(class_names)

# %%
"""
Get number of classes
"""

# %%
num_classes = len(class_names)
print(num_classes)

# %%
# Hight and width of the images
IMAGE_SIZE = 32
# 3 channels, Red, Green and Blue
CHANNELS = 3

# %%
"""
### Fetch and decode data
"""

# %%
"""
Load the training dataset. Labels are integers whereas class is one-hot encoded vectors.
"""

# %%
images_train, labels_train, class_train = get_train_data()

# %%
"""
Normal labels
"""

# %%
print(labels_train)

# %%
"""
One hot encoded labels
"""

# %%
print(class_train)

# %%
"""
Load the testing dataset.
"""

# %%
images_test, labels_test, class_test = get_test_data()

# %%
print("Training set size:\t",len(images_train))
print("Testing set size:\t",len(images_test))

# %%
"""
The CIFAR-10 dataset has been loaded and consists of a total of 60,000 images and corresponding labels.
"""

# %%
"""
### Define the CNN model
"""

# %%
def cnn_model():
    
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS)))    
    model.add(Conv2D(32, (3, 3), activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.summary()
    
    return model

# %%
"""
Build model
"""

# %%
model = cnn_model()

# %%
"""
### Train model on the training data
"""

# %%
"""
Save the model after every epoch
"""

# %%
checkpoint = ModelCheckpoint('best_model_simple.h5',  # model filename
                             monitor='val_loss', # quantity to monitor
                             verbose=0, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='auto') # The decision to overwrite model is made 
                                          # automatically depending on the quantity to monitor 

# %%
"""
Configure the model for training
"""

# %%
model.compile(loss='categorical_crossentropy', # Better loss function for neural networks
              optimizer=Adam(lr=1.0e-4), # Adam optimizer with 1.0e-4 learning rate
              metrics = ['accuracy']) # Metrics to be evaluated by the model

# %%
"""
For more information on categorical cross entropy loss function see - https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/
"""

# %%
"""
Fit the model on the data provided
"""

# %%
model_details = model.fit(images_train, class_train,
                    batch_size = 128, # number of samples per gradient update
                    epochs = 100, # number of iterations
                    validation_data= (images_test, class_test),
                    callbacks=[checkpoint],
                    verbose=1)

# %%
"""
### Evaluate the model
"""

# %%
scores = model.evaluate(images_test, class_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# %%
"""
### Model accuracy and loss plots
"""

# %%
plot_model(model_details)

# %%
"""
### Predictions
"""

# %%
"""
Predict class for test set images
"""

# %%
class_pred = model.predict(images_test, batch_size=32)
print(class_pred[0])

# %%
"""
Get the index of the largest element in each vector
"""

# %%
labels_pred = np.argmax(class_pred,axis=1)
print(labels_pred)

# %%
"""
Check which labels have been predicted correctly
"""

# %%
correct = (labels_pred == labels_test)
print(correct)
print("Number of correct predictions: %d" % sum(correct))

# %%
"""
Calculate accuracy using manual calculation
"""

# %%
num_images = len(correct)
print("Accuracy: %.2f%%" % ((sum(correct)*100)/num_images))

# %%
"""
### Show some mis-classifications
"""

# %%
"""
Get the incorrectly classified images
"""

# %%
incorrect = (correct == False)

# Images of the test-set that have been incorrectly classified.
images_error = images_test[incorrect]

# Get predicted classes for those images
labels_error = labels_pred[incorrect]

# Get true classes for those images
labels_true = labels_test[incorrect]

# %%
"""
Plot the first 9 mis-classified images
"""

# %%


# %%
"""

"""