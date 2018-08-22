import librosa
import librosa.display
import glob
import random
import keras
from keras.models import Sequential
from keras.layers import RNN, LSTM, Dense, Flatten, Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling3D
import numpy as np

dataset = []
count = 0

#load all the audio files from a directory into teh list
list_of_audio_files = [f for f in glob.glob('/Users/User/Downloads/free-spoken-digit-dataset-master/recordings/*.wav')]

for audio_file in list_of_audio_files:
    count += 1

    #set a
    if count%100 == 0:
        print('done loading',count)
    #loading the audio file as time series and extracting its sampling rate
    data,sampling_rate = librosa.load(audio_file)

    #extract mfcc features form the extracted time series
    feature = librosa.feature.mfcc(y = data,sr = sampling_rate)

    #reshaoe the 2D matrix to 20x30
    padded_feature = librosa.util.fix_length(feature,30)

    #get the value
    fields = audio_file.split('/')
    temp = fields[len(fields)-1]
    name = temp.split('_')
    if name[1] == 'jackson':
        target = '1'
        dataset.append((padded_feature, target))
    if name[1] == 'nicolas':
        target = '0'
        dataset.append((padded_feature,target))

random.shuffle(dataset)

train = dataset[:240]
test = dataset[240:]

x_train,y_train = zip(*train)
x_test,y_test = zip(*test)

x_train = np.array([x.reshape( (20, 30) ) for x in x_train])
x_test = np.array([x.reshape( (20, 30) ) for x in x_test])


y_train = np.array(keras.utils.to_categorical(y_train,num_classes=None))
y_test = np.array(keras.utils.to_categorical(y_test,num_classes=None))

model = Sequential()


# works
model.add(LSTM(128,activation='sigmoid'))
model.add(Dense(300,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='softmax'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))


model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10,batch_size=50)
acc = model.evaluate(x_test, y_test,batch_size=50)

print('Test accuracy:', acc)
predicted = model.predict(x_test,verbose=0)
#uncomment it if you want to print the output
'''
for p in predicted:
    if p[0] > p[1]:
        print ('0')
    else:
        print ('1')
'''

print(model.summary())

