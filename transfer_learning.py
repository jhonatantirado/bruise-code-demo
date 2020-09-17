from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import bruise_network as network_v1

num_classes = 7
bruise_width = 224
bruise_height = 224
bruise_channels = 3
num_epochs = 100
batch_size = 32
dense_layer_neurons = 256
split_layers_index = 0
training_data_path = './equimosisv3/Training/'
testing_data_path = './Dataset/Test/'
image_path_test = './test-fruits/20191010_070501_Dia4_SixDays.jpg'

inception_v3_model_name = 'bruise_inceptionv3'
resnet50_model_name = 'bruise_resnet50'
mobilenet_model_name = 'bruise_mobilenet'
network_v1_model_name = 'bruise_network_v1'

date_suffix = '_060220'
model_version = '_v1'
model_extension = '.h5'

checkpoint_filepath = inception_v3_model_name + '_weights.best' + date_suffix + model_version + '.hdf5'

training_mode = True
resume_training_mode = False
prediction_mode = False

validation_split = 0.1

period_checkpoint = 5
initial_learning_rate = 1e-1
momentum = 0.9
lr_update_factor = 0.1

with open('labels') as f:
    labels = f.readlines()
labels = [x.strip() for x in labels]


def get_model(type=0):
    if type == 0:
        from keras.applications.inception_v3 import InceptionV3, preprocess_input
        model = InceptionV3(weights='imagenet', include_top=False,
                            input_shape=(bruise_width, bruise_height, bruise_channels))
        func = preprocess_input

    elif type == 1:
        from keras.applications.resnet50 import ResNet50, preprocess_input
        model = ResNet50(weights='imagenet', include_top=False,
                         input_shape=(bruise_width, bruise_height, bruise_channels))
        func = preprocess_input

    elif type == 2:
        from keras.applications.mobilenet import MobileNet, preprocess_input
        model = MobileNet(weights='imagenet', include_top=False,
                          input_shape=(bruise_width, bruise_height, bruise_channels))
        func = preprocess_input

    elif type == 3:
        from keras.applications.inception_v3 import preprocess_input
        model = network_v1.conv_net()
        func = preprocess_input

    return model, preprocess_input


def build_training_validation_generator(data_path, preprocess_input):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=validation_split)

    train_generator = datagen.flow_from_directory(data_path,
                                                  target_size=(bruise_width, bruise_height),
                                                  color_mode='rgb',
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  subset='training')

    validation_generator = datagen.flow_from_directory(data_path,
                                                       target_size=(bruise_width, bruise_height),
                                                       color_mode='rgb',
                                                       batch_size=batch_size,
                                                       class_mode='categorical',
                                                       shuffle=True,
                                                       subset='validation')
    return train_generator, validation_generator


def build_model(num_classes, base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_layer_neurons, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in model.layers[:split_layers_index]:
        layer.trainable = False
    for layer in model.layers[split_layers_index:]:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=initial_learning_rate, momentum=momentum, decay=initial_learning_rate / num_epochs),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_callbacks(checkpoint_filepath, period_checkpoint):
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto',
                                 period=period_checkpoint)
    return [checkpoint]


def retrain(new_model, step_size_train, train_generator, validation_generator, step_size_validate):
    startTime = datetime.now()
    print('[INFO] Start training: ' + str(startTime))

    callbacks_list = get_callbacks(checkpoint_filepath, period_checkpoint)

    if resume_training_mode and checkpoint_filepath is not None and os.path.exists(checkpoint_filepath):
        print('[INFO] Resume training from: ' + checkpoint_filepath)
        model = load_model(checkpoint_filepath)
        print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
        old_lr = K.get_value(model.optimizer.lr)
        new_lr = old_lr * lr_update_factor
        K.set_value(model.optimizer.lr, new_lr)
        print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))
    else:
        model = new_model

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=step_size_train,
                        epochs=num_epochs,
                        validation_data=validation_generator,
                        validation_steps=step_size_validate,
                        callbacks=callbacks_list, verbose=1)

    endTime = datetime.now()
    duration = endTime - startTime
    print('[INFO] End training: ' + str(endTime))
    print('[INFO] Training time: ' + str(duration))

    return model


def load_image_for_prediction(image_path, preprocess_input):
    img = image.load_img(image_path, target_size=(bruise_width, bruise_height, bruise_channels))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img


def load_model_for_prediction(model_path):
    model = load_model(model_path)

    return model


def transfer_learning_and_save_model(model_index, model_name):
    if os.path.exists(model_name) and not resume_training_mode:
        print('[INFO] A model with the same name already exists: ' + model_name
              + '. Please change date_suffix, model_version or use a different architecture')
        sys.exit()

    print('[INFO] Transfer learning with ' + model_name)
    model, preprocess_function = get_model(model_index)
    train_generator, validation_generator = build_training_validation_generator(training_data_path, preprocess_function)
    step_size_train = train_generator.n // train_generator.batch_size
    step_size_validate = validation_generator.n // validation_generator.batch_size
    new_model = build_model(num_classes, model)
    new_model = retrain(new_model, step_size_train, train_generator, validation_generator, step_size_validate)
    new_model.save(model_name)

    visualize_metrics(new_model.history, model_name)


def predict(model_index, model_name, image_path):
    print('[INFO] Prediction with ' + model_name)
    _, preprocess_function = get_model(model_index)
    img = load_image_for_prediction(image_path, preprocess_function)
    loaded_model = load_model_for_prediction(model_name)
    classes = loaded_model.predict(img)
    return classes


def visualize_metrics(history, model_name):
    print('[INFO] Max training accuracy: ' + str(max(history.history['acc'])))
    print('[INFO] Max validation accuracy: ' + str(max(history.history['val_acc'])))

    print('[INFO] Min training loss: ' + str(min(history.history['loss'])))
    print('[INFO] Min validation loss: ' + str(min(history.history['val_loss'])))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(model_name + '_Accuracy.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(model_name + '_Loss.png')


if __name__ == '__main__':

    model_file_suffix = date_suffix + model_version + model_extension
    print('[INFO] Epochs: ' + str(num_epochs))
    print('[INFO] Batch size: ' + str(batch_size))
    print('[INFO] Classes: ' + str(num_classes))
    print('[INFO] Dense layer neurons: ' + str(dense_layer_neurons))
    print('[INFO] Bruise width: ' + str(bruise_width))
    print('[INFO] Bruise height: ' + str(bruise_height))
    print('[INFO] Bruise channels: ' + str(bruise_channels))
    print('[INFO] Learning rate: ' + str(initial_learning_rate))
    print('[INFO] Retrain layers: ' + str(split_layers_index))
    print('[INFO] Validation split: ' + str(validation_split))
    print('[INFO] Period checkpoint: ' + str(period_checkpoint))
    print('[INFO] Training mode: ' + str(training_mode))
    print('[INFO] Resume Training mode: ' + str(resume_training_mode))
    print('[INFO] Prediction mode: ' + str(prediction_mode))
    print('[INFO] Training data: ' + str(training_data_path))
    print('[INFO] Test data: ' + str(testing_data_path))
    print('[INFO] Test image path: ' + str(image_path_test))

    if training_mode:
        # transfer_learning_and_save_model(0, inception_v3_model_name + model_file_suffix)
        # transfer_learning_and_save_model(1, resnet50_model_name + model_file_suffix)
        transfer_learning_and_save_model(2, mobilenet_model_name + model_file_suffix)

    if prediction_mode:
        prediction = predict(0, inception_v3_model_name + model_file_suffix, image_path_test)
        print(prediction, np.argmax(prediction), labels[np.argmax(prediction)])

        '''
        prediction = predict(0, resnet50_model_name + model_file_suffix, image_path_test)
        print (prediction)

        prediction = predict(0, mobilenet_model_name + model_file_suffix, image_path_test)
        print (prediction)
        '''
