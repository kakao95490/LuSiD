from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D, Dense

def create_model(image_shape, n1, f1, n2, f2, n3, f3):
    model = Sequential()
    
    model.add(Conv2D(n1, f1, activation='relu', input_shape=image_shape))
    
    model.add(Conv2D(n2, f2, activation='relu'))

    # Upsampling   (double la taille de l'image, 128x128 -> 256x256 par exemple)
    # model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(n3, f3, activation='sigmoid'))
    
    return model
