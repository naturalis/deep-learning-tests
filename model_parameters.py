#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 09:41:55 2019

@author: maarten
"""
















class modelParameteres:
  
  
  model_architecture = 
  Xception
  ResNet50
  InceptionV3
  DenseNet121




Documentation for individual models
Model	Size	Top-1 Accuracy	Top-5 Accuracy	Parameters	Depth
Xception	88 MB	0.790	0.945	22,910,480	126
VGG16	528 MB	0.713	0.901	138,357,544	23
VGG19	549 MB	0.713	0.900	143,667,240	26
ResNet50	99 MB	0.749	0.921	25,636,712	168
InceptionV3	92 MB	0.779	0.937	23,851,784	159
InceptionResNetV2	215 MB	0.803	0.953	55,873,736	572
MobileNet	16 MB	0.704	0.895	4,253,864	88
MobileNetV2	14 MB	0.713	0.901	3,538,984	88
DenseNet121	33 MB	0.750	0.923	8,062,504	121
DenseNet169	57 MB	0.762	0.932	14,307,880	169
DenseNet201	80 MB	0.773	0.936	20,242,984	201
NASNetMobile	23 MB	0.744	0.919	5,326,716	-
NASNetLarge	343 MB	0.825	0.960	88,949,818	-





# architecture
# weights 'imagenet' / None
# input shape (150, 150, 3)
# activation='softmax')
#        model.compile(loss='binary_crossentropy',
#        optimizer=optimizers.RMSprop(lr=1e-5),
#        metrics=['acc'])
#datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)      
#        target_size=(150,150), 
#        batch_size=32,
#        interpolation="nearest",
#    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
#
#    checkpoint = ModelCheckpoint(filepath=self.modelRepoFolder + self.modelName + '.hd5', 
#                                 verbose=1, 
#                                 save_best_only=True, 
#                                 monitor='val_loss', 
#                                 mode='auto')

dataset also include class lists
