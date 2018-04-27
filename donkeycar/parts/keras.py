'''

pilots.py

Methods to create, use, save and load pilots. Pilots
contain the highlevel logic used to determine the angle
and throttle of a vehicle. Pilots can include one or more
models to help direct the vehicles motion.

'''




import os
import numpy as np
import keras

import donkeycar as dk


class KerasPilot():

    def load(self, model_path):
        self.model = keras.models.load_model(model_path)


    def shutdown(self):
        pass

    def print_summary(self):
        self.model.summary()


    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):

        """
        train_gen: generator that yields an array of images an array of

        """

        #checkpoint to save model after each epoch
        save_best = keras.callbacks.ModelCheckpoint(saved_model_path,
                                                    monitor='val_loss',
                                                    verbose=verbose,
                                                    save_best_only=True,
                                                    mode='min')

        #stop training if the validation error stops improving.
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=min_delta,
                                                   patience=patience,
                                                   verbose=verbose,
                                                   mode='auto')

        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)

        hist = self.model.fit_generator(
                        train_gen,
                        steps_per_epoch=steps,
                        epochs=epochs,
                        verbose=1,
                        validation_data=val_gen,
                        callbacks=callbacks_list,
                        validation_steps=steps*(1.0 - train_split))
        return hist


class KerasCategorical(KerasPilot):
    def __init__(self, model=None, alternate=False, *args, **kwargs):
        super(KerasCategorical, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        elif alternate:
            self.model = categorical_alternate()
        else:
            self.model = default_categorical()
        if constant_throttle[0]:
            self.constant_throttle = constant_throttle

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, throttle = self.model.predict(img_arr)
        print('throttle: {}'.format(throttle))
        #angle_certainty = max(angle_binned[0])
        angle_unbinned = dk.utils.linear_unbin(angle_binned)
        print('Angle unbinned: {}, Throttle: {}'.format(angle_unbinned, throttle[0][0]))
        return angle_unbinned, throttle[0][0]



class KerasLinear(KerasPilot):
    def __init__(self, model=None, num_outputs=None, max_throttle=None, *args, **kwargs):
        super(KerasLinear, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        elif num_outputs is not None:
            self.model = default_n_linear(num_outputs)
        else:
            self.model = default_linear()
        self.max_throttle = max_throttle

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        #print("Angle: {}, Throttle: {}".format(outputs[0][0], outputs[1][0]))
        steering = outputs[0][0][0]
        # Cap the throttle if max_throttle is initialized
        throttle = 0.0
        if self.max_throttle is not None:
            if self.max_throttle < outputs[1][0][0]:
                throttle = self.max_throttle
            else:
                throttle = outputs[1][0][0]
        else:
            throttle = outputs[1][0][0]
#        print("Angle: {}, Throttle: {}".format(steering, throttle))
        return steering, throttle

class KerasCustom(KerasPilot):
    def __init__(self, model=None, num_outputs=None, throttle=0.0, *args, **kwargs):
        super(KerasCustom, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = custom_linear()
        self.throttle = throttle

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        #print("Angle: {}, Throttle: {}".format(outputs[0][0], outputs[1][0]))
        steering = outputs[0][0]
        throttle = self.throttle
        return steering, throttle

class NvidiaPilot(KerasPilot):
    def __init__(self, model=None, constant_throttle=0.0, *args, **kwargs):
        super(NvidiaPilot, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = nividia_linear()
        self.constant_throttle = constant_throttle

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        output = self.model.predict(img_arr)
        # print("Angle: {}".format(output[0][0]))
        steering = output[0][0]
        return steering, self.constant_throttle

class KerasIMU(KerasPilot):
    '''
    A Keras part that take an image and IMU vector as input,
    outputs steering and throttle

    Note: When training, you will need to vectorize the input from the IMU.
    Depending on the names you use for imu records, something like this will work:

    X_keys = ['cam/image_array','imu_array']
    y_keys = ['user/angle', 'user/throttle']

    def rt(rec):
        rec['imu_array'] = np.array([ rec['imu/acl_x'], rec['imu/acl_y'], rec['imu/acl_z'],
            rec['imu/gyr_x'], rec['imu/gyr_y'], rec['imu/gyr_z'], rec['imu/temp'] ])
        return rec

    kl = KerasIMU()

    tubgroup = TubGroup(tub_names)
    train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys, record_transform=rt,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    train_frac=cfg.TRAIN_TEST_SPLIT)

    '''
    def __init__(self, model=None, num_outputs=2, num_imu_inputs=7 , *args, **kwargs):
        super(KerasIMU, self).__init__(*args, **kwargs)
        self.num_imu_inputs = num_imu_inputs
        self.model = default_imu(num_outputs = num_outputs, num_imu_inputs = num_imu_inputs)

    def run(self, img_arr, accel_x, accel_y, accel_z, gyr_x, gyr_y, gyr_z, temp):
        #TODO: would be nice to take a vector input array.
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        imu_arr = np.array([accel_x, accel_y, accel_z, gyr_x, gyr_y, gyr_z, temp]).reshape(1,self.num_imu_inputs)
        outputs = self.model.predict([img_arr, imu_arr])
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]

class KerasBehavioral(KerasPilot):
    '''
    A Keras part that take an image and Behavior vector as input,
    outputs steering and throttle
    '''
    def __init__(self, model=None, num_outputs=2, num_behavior_inputs=2, input_shape=(120,160,3), *args, **kwargs):
        super(KerasBehavioral, self).__init__(*args, **kwargs)
        self.model = default_bhv(num_outputs = num_outputs, num_bvh_inputs = num_behavior_inputs, input_shape=input_shape)
        self.compile()

    def compile(self):
        self.model.compile(optimizer='adam',
                  loss='mse')
        
    def run(self, img_arr, state_array):        
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        bhv_arr = np.array(state_array).reshape(1,len(state_array))
        #angle_binned, throttle = self.model.predict([img_arr, bhv_arr])
        angle, throttle = self.model.predict([img_arr, bhv_arr])
        #in order to support older models with linear throttle,
        #we will test for shape of throttle to see if it's the newer
        #binned version.
        # N = len(throttle[0])
        
        # if N > 0:
        #     throttle = dk.utils.linear_unbin(throttle, N=N, offset=0.0, R=0.5)
        # else:
        #     throttle = throttle[0][0]
        # angle_unbinned = dk.utils.linear_unbin(angle_binned)
        return angle[0], throttle[0]

def default_bhv(num_outputs, num_bvh_inputs, input_shape):
    '''
    Notes: this model depends on concatenate which failed on keras < 2.0.8
    '''

    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
    from keras.layers.merge import concatenate
    
    img_in = Input(shape=input_shape, name='img_in')
    bvh_in = Input(shape=(num_bvh_inputs,), name="behavior_in")
    
    x = img_in
    #x = Cropping2D(cropping=((60,0), (0,0)))(x) #trim 60 pixels off top
    #x = Lambda(lambda x: x/127.5 - 1.)(x) # normalize and re-center
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)
    
    y = bvh_in
    y = Dense(num_bvh_inputs * 2, activation='relu')(y)
    y = Dense(num_bvh_inputs * 2, activation='relu')(y)
    y = Dense(num_bvh_inputs * 2, activation='relu')(y)
    
    z = concatenate([x, y])
    z = Dense(100, activation='relu')(z)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)

    #Continuous output of throttle
    angle_out = Dense(1, activation='linear', name='angle_out')(z)
    
    #continous output of throttle
    throttle_out = Dense(1, activation='linear', name='throttle_out')(z)      # Reduce to 1 number, Positive number only
        
    model = Model(inputs=[img_in, bvh_in], outputs=[angle_out, throttle_out])
    
    return model

def default_categorical():
    from keras.layers import Input, Dense, merge
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Dense

    img_in = Input(shape=(120, 160, 3), name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)       # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)       # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)                                        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)                                    # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)                                                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)                                     # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)                                                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    #categorical output of the angle
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    #continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)      # Reduce to 1 number, Positive number only

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy',
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .001})

    return model

def default_linear():
    from keras.layers import Input, Dense, merge
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Dense

    img_in = Input(shape=(120,160,3), name='img_in')
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='linear')(x)
    x = Dropout(.1)(x)
    x = Dense(50, activation='linear')(x)
    x = Dropout(.1)(x)
    #categorical output of the angle
    angle_out = Dense(1, activation='linear', name='angle_out')(x)

    #continous output of throttle
    throttle_out = Dense(1, activation='linear', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])


    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': .5})

    return model


def nividia_linear():
    from keras.optimizers import Adam
    from keras.layers import Input, Dense, merge
    from keras.models import Model, Sequential
    from keras.layers import Conv2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Flatten, Dense
    adam = Adam(lr=0.0001)
    model = Sequential()


    model.add(BatchNormalization(input_shape=(120,160,3), epsilon=0.001, axis=1))
    model.add(Conv2D(24,(5,5),padding='valid', activation='relu', strides=(2,2)))
    model.add(Conv2D(36,(5,5),padding='valid', activation='relu', strides=(2,2)))
    model.add(Conv2D(48,(5,5),padding='valid', activation='relu', strides=(2,2)))
    model.add(Conv2D(64,(3,3),padding='valid', activation='relu', strides=(1,1)))
    model.add(Conv2D(64,(3,3),padding='valid', activation='relu', strides=(1,1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    model.compile(loss='mse',
              optimizer=adam,
              metrics=['mse','accuracy'])
    return model


def custom_linear():
    from keras.optimizers import Adam
    from keras.layers import Input, Dense
    from keras.models import Model, Sequential
    from keras.layers import Conv2D, BatchNormalization
    from keras.layers import Flatten, Dense, Dropout

    adam = Adam(lr=0.0001)
    model = Sequential()
    model.add(BatchNormalization(input_shape=(120,160,3), epsilon=0.001, axis=1))
    model.add(Conv2D(24, (5,5), padding='valid', activation='relu', strides=(2,2)))
    model.add(Conv2D(32, (5,5), padding='valid', activation='relu', strides=(2,2)))
    model.add(Conv2D(64, (5,5), padding='valid', strides=(2,2), activation='relu'))
    model.add(Conv2D(64, (3,3), strides=(1,1), padding='valid', activation='relu'))
    model.add(Conv2D(64, (3,3), strides=(1,1), padding='valid', activation='relu'))

    model.add(Flatten(name='flattened'))
    model.add(Dense(150,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='tanh',name='angle_out'))

    model.compile(loss='mse',
              optimizer=adam,
              metrics=['mse','accuracy'])
    return model


def default_n_linear(num_outputs):
    from keras.layers import Input, Dense, merge
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda

    img_in = Input(shape=(120,160,3), name='img_in')
    x = img_in
    x = Cropping2D(cropping=((60,0), (0,0)))(x) #trim 60 pixels off top
    x = Lambda(lambda x: x/127.5 - 1.)(x) # normalize and re-center
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (5,5), strides=(1,1), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(.1)(x)

    outputs = []

    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in], outputs=outputs)


    model.compile(optimizer='adam',
                  loss='mse')

    return model



def default_imu(num_outputs, num_imu_inputs):
    '''
    Notes: this model depends on concatenate which failed on keras < 2.0.8
    '''

    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
    from keras.layers.merge import concatenate

    img_in = Input(shape=(120,160,3), name='img_in')
    imu_in = Input(shape=(num_imu_inputs,), name="imu_in")

    x = img_in
    x = Cropping2D(cropping=((60,0), (0,0)))(x) #trim 60 pixels off top
    #x = Lambda(lambda x: x/127.5 - 1.)(x) # normalize and re-center
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)

    y = imu_in
    y = Dense(14, activation='relu')(y)
    y = Dense(14, activation='relu')(y)
    y = Dense(14, activation='relu')(y)

    z = concatenate([x, y])
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)

    outputs = []

    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='out_' + str(i))(z))

    model = Model(inputs=[img_in, imu_in], outputs=outputs)

    model.compile(optimizer='adam',
                  loss='mse')

    return model
