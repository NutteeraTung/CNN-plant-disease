#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from time import perf_counter 
import os


# In[2]:


batch_size = 100
img_height = 250
img_width = 250


# In[4]:


## loading training set
training_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'D:\sample 1\corn disease\train',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size

)


# In[7]:


## loading validation dataset
validation_ds =  tf.keras.preprocessing.image_dataset_from_directory(
    r'D:\sample 1\corn disease\valid',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size)


# In[8]:


class_names = training_ds.class_names


# In[9]:


## Defining Cnn
MyCnn = tf.keras.models.Sequential([
  layers.BatchNormalization(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dense(len(class_names), activation= 'softmax')
])


# In[10]:


MyCnn.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[11]:


## lets train our CNN
retVal = MyCnn.fit(training_ds,validation_data= validation_ds,epochs = 5)


# In[13]:


plt.plot(retVal.history['loss'], label = 'training loss')
plt.plot(retVal.history['accuracy'], label = 'training accuracy')
plt.legend()


# In[14]:


AccuracyVector = []
plt.figure(figsize=(30, 30))
for images, labels in validation_ds.take(1):
    predictions = MyCnn.predict(images)
    predlabel = []
    prdlbl = []
    
    for mem in predictions:
        predlabel.append(class_names[np.argmax(mem)])
        prdlbl.append(np.argmax(mem))
    
    AccuracyVector = np.array(prdlbl) == labels
    for i in range(40):
        ax = plt.subplot(10, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title('Pred: '+ predlabel[i]+' actl:'+class_names[labels[i]] )
        plt.axis('off')
        plt.grid(True)


# In[15]:


plt.plot(retVal.history['val_loss'], label = 'validation loss')
plt.plot(retVal.history['val_accuracy'], label = 'validation accuracy')
plt.legend()


# In[19]:


MyCnn.save('model.h5')


# In[20]:


#####################################################################
# lets now see time taken and validation accuracy and model size    #
#####################################################################
start = perf_counter() 
_, unPrunnedAccuracy = MyCnn.evaluate(validation_ds, verbose = 0 )
end = perf_counter() 

sizeUnprunned = os.path.getsize('model.h5')
# let convert to MB
sizeUnprunned = sizeUnprunned / (1024 * 1024)

print ('unPrunned model Summary:')
print('Model size(MB) : {}'.format(sizeUnprunned))
print('Time on Validation data (sec) : {}'.format(end - start))
print('Accuracy on validation data: {}'.format(unPrunnedAccuracy))


# In[21]:


########################################################################
# lets now implement weight prunning                                   #
########################################################################
get_ipython().system('pip install -q tensorflow-model-optimization')
import tensorflow_model_optimization as tfmot
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

epochs = 5
end_step = np.ceil(70295 / batch_size).astype(np.int32) * epochs
## pruning param
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.80, begin_step=0, end_step=end_step)
                }
## defining prunned model
MyPrunnedModel = prune_low_magnitude(MyCnn, **pruning_params)
MyPrunnedModel.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[22]:


callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir='.'),
]
retVal = MyPrunnedModel.fit(training_ds,validation_data= validation_ds,epochs = 5 ,callbacks= callbacks)


# In[23]:


#################################################################
# let Benmark this as above                                     #
#################################################################
plt.plot(retVal.history['val_loss'], label = 'validation loss')
plt.plot(retVal.history['val_accuracy'], label = 'validation accuracy')
plt.legend()


# In[24]:


plt.plot(retVal.history['loss'], label = 'training loss')
plt.plot(retVal.history['accuracy'], label = 'training accuracy')
plt.legend()


# In[25]:


MyPrunnedModel.save('Prunnedmodel.h5')


# In[26]:


start = perf_counter() 
_, PrunnedAccuracy = MyPrunnedModel.evaluate(validation_ds, verbose = 0 )
end = perf_counter() 

sizePrunned = os.path.getsize('Prunnedmodel.h5')
# let convert to MB
sizePrunned = sizePrunned / (1024 * 1024)

print ('Prunned model Summary:')
print('Model size(MB) : {}'.format(sizePrunned))
print('Time on Validation data (sec) : {}'.format(end - start))
print('Accuracy on validation data: {}'.format(PrunnedAccuracy))


# In[ ]:




