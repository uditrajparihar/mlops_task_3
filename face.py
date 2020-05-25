#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications import resnet50


# In[2]:


model = resnet50.ResNet50(weights = 'imagenet' , 
                    include_top = False ,
                    input_shape = (224 , 224 , 3) )


# In[3]:


for layer in (model.layers):
    layer.trainable = False 


# In[4]:


for layer in (model.layers):
    print(layer.__class__.__name__ , layer.trainable)


# In[5]:


model.output


# In[6]:


from keras.models import Sequential


# In[7]:


from keras.layers import Dense


# In[8]:


from keras.layers import Flatten


# In[9]:


patch_model = model.output


# In[10]:


patch_model = Flatten()(patch_model)


# In[11]:


patch_model = Dense(units=500 , activation='relu')(patch_model)


# In[12]:


patch_model = Dense(units=100 , activation='relu')(patch_model)


# In[13]:


patch_model = Dense(units=4 , activation='softmax')(patch_model)


# In[14]:


from keras.models import Model


# In[15]:


model.input


# In[16]:


patch_model


# In[17]:


newmodel = Model(inputs= model.input , outputs =patch_model )


# In[18]:


newmodel.summary()


# In[19]:



from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'faceRecog_ACNN/train_data'
validation_data_dir = 'faceRecog_ACNN/validation_data'

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# Change the batchsize according to your system RAM
train_batchsize = 16
val_batchsize = 10
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(224, 224),
        batch_size=train_batchsize,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(224, 224),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)


# In[20]:


from keras.optimizers import Adam
newmodel.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(lr = 0.01),
              metrics = ['accuracy'])


# In[21]:


newmodel.fit_generator(train_generator , validation_data=validation_generator , epochs=1)


# In[22]:


newmodel.save('Fam_FACE_RECOG_ResNet50.h5')


# In[23]:


from keras.preprocessing import image


# In[24]:


utest = image.load_img('udit6.jpeg' , target_size=(224 , 224 , 3) )

utest#.crop((95 ,40 ,130 , 100))
# In[25]:


u_test_np = image.img_to_array(utest)


# In[26]:


u_test_np.shape


# In[ ]:





# In[27]:


import numpy as np 


# In[28]:


u_test_np = np.expand_dims(u_test_np , axis = 0)


# In[29]:


u_test_np.shape


# In[30]:


pred = newmodel.predict(u_test_np )


# In[31]:


from keras.applications.resnet50 import preprocess_input 


# In[32]:


from keras.applications.resnet50 import decode_predictions


# In[33]:


pred


# In[34]:


train_generator.class_indices


# In[1]:





# In[ ]:





# In[ ]:




