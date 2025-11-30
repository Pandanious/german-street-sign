
from keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten, Dropout, Activation
from keras import Model

def GTSRB_model(num_classes, imwidth,imheight):

    gtsrb_input = Input(shape=(imwidth,imheight,3))

    x = Conv2D(32, (3,3), activation='relu')(gtsrb_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    #x = GlobalAvgPool2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=gtsrb_input, outputs=x)

def TSCNN_model(num_classes, imwidth,imheight):

    tscnn_input = Input(shape=(imwidth,imheight,3))
    x = Conv2D(128, (5,5), activation='relu')(tscnn_input)
    x = Conv2D(128, (5,5), activation='relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=tscnn_input, outputs=x)

    
def LTSNet_model(num_classes, imwidth,imheight):

    ltsnet_input = Input(shape=(imwidth,imheight,3))
    x = Conv2D(32, (5,5), use_bias = False)(ltsnet_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)

    x = Conv2D(64, (5,5), use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.15)(x)
    
    x = Conv2D(32, (5,5), use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.15)(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)
    
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)
    
    x = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=ltsnet_input, outputs=x)
    
    
    
    
    

if __name__=='__main__':

    model = GTSRB_model(10,60,60)
    model.summary()
