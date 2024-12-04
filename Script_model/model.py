from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model(image_shape,n1,f1,n2,f2,n3,f3):
    model = Sequential()

    model.add(Conv2D(n1,f1,activation='relu',input_shape=image_shape))

    model.add(Conv2D(n2,f2,activation='relu'))

    model.add(Conv2D(n3, f3, activation=None))

    return model
