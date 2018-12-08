# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:50:43 2018

@author: jaley
"""

from keras.models import Sequential
from keras.layers import Dense;

model = Sequential()
model.add(Dense(units=3,activation='relu'))
model.add(Dense(units=5,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
print (model.get_config())