from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import csv
import pandas as pd
import shutil
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# number of classes
num_classes = 7
bruise_width = 224
bruise_height = 224
bruise_channels = 3
num_epochs = 1
batch_size = 32
dense_layer_neurons = 256
split_layers_index = -3
training_data_path = './equimosisv3/Training/'
testing_data_path = './equimosisv3/Test/'
image_path_test = './test-fruits/20191010_070501_Dia4_SixDays.jpg'
training_data_path_k_fold = "./k-fold-cv/training"
validation_data_path_k_fold = "./k-fold-cv/validation"
testing_data_path_k_fold = "./k-fold-cv/test"

inception_v3_model_name = 'bruise_inceptionv3'
resnet50_model_name = 'bruise_resnet50'
mobilenet_model_name = 'bruise_mobilenet'

date_suffix = '_090220'
model_version = '_v1'
model_extension = '.h5'

checkpoint_filepath="weights.best.hdf5"

training_mode = False
resume_training_mode = False
prediction_mode = False
generate_k_fold_list_mode = False
predict_k_fold_mode = False

k_fold_cross_validation_mode = True

validation_split = 0.1

# k-folds cross validation
n_folds = 10

with open(os.getcwd()  + '\\labels') as f:
    labels = f.readlines()
labels = [x.strip() for x in labels]

# https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
def get_base_model(type = 0):
    if type == 0:
        from keras.applications.inception_v3 import InceptionV3,preprocess_input
        model = InceptionV3(weights='imagenet', include_top=False, input_shape=(bruise_width,bruise_height,bruise_channels))
        func = preprocess_input

    elif type == 1:
        from keras.applications.resnet50 import ResNet50,preprocess_input
        model = ResNet50(weights='imagenet', include_top=False, input_shape=(bruise_width,bruise_height,bruise_channels))
        func = preprocess_input

    elif type == 2:
        from keras.applications.mobilenet import MobileNet,preprocess_input
        model = MobileNet(weights='imagenet', include_top=False, input_shape=(bruise_width,bruise_height,bruise_channels))
        func = preprocess_input

    return model, preprocess_input

def build_training_validation_generator(data_path, preprocess_input):
    datagen = ImageDataGenerator(preprocessing_function = preprocess_input,validation_split= validation_split)

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

def build_generator(data_path, preprocess_input, batch_size = 1):
    datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

    generator = datagen.flow_from_directory(data_path,
                                                     target_size=(bruise_width, bruise_height),
                                                     color_mode='rgb',
                                                     batch_size=batch_size,
                                                     class_mode='categorical',
                                                     shuffle=True)

    return generator

def get_init_epoch(checkpoint_filepath):
    return 0

def get_callbacks(checkpoint_filepath, patience_lr, patience_early_stopping):
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=1)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience_early_stopping, verbose=1, mode='auto', restore_best_weights=False)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='auto')
    return [checkpoint, early_stopping, reduce_lr_loss]

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

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def retrain(new_model, step_size_train, train_generator, validation_generator, step_size_validate):
    callbacks_list = get_callbacks(checkpoint_filepath, 3, 3)
    initial_epoch = 0

    if resume_training_mode and checkpoint_filepath is not None and os.path.exists(checkpoint_filepath):
        model = load_model(checkpoint_filepath)
        initial_epoch = get_init_epoch(checkpoint_filepath)
    else:
        model = new_model

    model.fit_generator(generator=train_generator,
                       steps_per_epoch=step_size_train,
                       epochs=num_epochs,
                       validation_data = validation_generator,
                       validation_steps = step_size_validate,
                       callbacks=callbacks_list, verbose=1, initial_epoch=initial_epoch)

    return model

def retrain_k_fold(new_model, step_size_train, train_generator, validation_generator, step_size_validate):
    callbacks_list = get_callbacks(checkpoint_filepath, 3, 3)
    initial_epoch = 0

    if resume_training_mode and checkpoint_filepath is not None and os.path.exists(checkpoint_filepath):
        model = load_model(checkpoint_filepath)
        initial_epoch = get_init_epoch(checkpoint_filepath)
    else:
        model = new_model

    history = model.fit_generator(generator=train_generator,
                       steps_per_epoch=step_size_train,
                       epochs=num_epochs,
                       validation_data = validation_generator,
                       validation_steps = step_size_validate,
                       callbacks=callbacks_list, verbose=1, initial_epoch=initial_epoch)

    return model, history

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
    print ('Transfer learning with ' + model_name)
    model, preprocess_function = get_base_model(model_index)
    train_generator, validation_generator = build_training_validation_generator(training_data_path, preprocess_function)
    step_size_train = train_generator.n//train_generator.batch_size
    step_size_validate = validation_generator.n//validation_generator.batch_size
    new_model = build_model(num_classes, model)
    new_model = retrain(new_model, step_size_train, train_generator, validation_generator, step_size_validate)
    new_model.save(model_name)

def transfer_learning_and_save_model_k_fold(model_index, model_name, k_fold = 1):
    model_name =  str(k_fold) + '_' + model_name
    print ('Transfer learning with ' + model_name)
    model, preprocess_function = get_base_model(model_index)
    train_generator = build_generator(training_data_path_k_fold, preprocess_function, batch_size)
    validation_generator = build_generator(validation_data_path_k_fold, preprocess_function, batch_size)
    step_size_train = train_generator.n//train_generator.batch_size
    step_size_validate = validation_generator.n//validation_generator.batch_size
    new_model = build_model(num_classes, model)
    new_model, history = retrain_k_fold(new_model, step_size_train, train_generator, validation_generator, step_size_validate)

    new_model.save(model_name)

    return new_model, history, model_name

def predict(model_index, model_name, image_path):
    print ('Prediction with ' + model_name)
    _, preprocess_function = get_base_model(model_index)
    img = load_image_for_prediction(image_path, preprocess_function)
    loaded_model = load_model_for_prediction(model_name)
    classes = loaded_model.predict(img)
    return classes

def predict_k_fold(model_index, model_name):
    print ('Prediction with ' + model_name)
    _, preprocess_function = get_base_model(model_index)
    model = load_model_for_prediction(model_name)
    test_generator = build_generator(testing_data_path_k_fold, preprocess_function)
    predictions = model.predict_generator(test_generator, steps = test_generator.n)
    return predictions

def generate_k_fold_list(rootdir, output_csv_file_name):
    with open(output_csv_file_name, mode='w', newline='') as training_file:
        training_writer = csv.writer(training_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        training_writer.writerow(["file_name", "file_full_path", "file_class","k_fold"])
        for subdir, dirs, files in os.walk(rootdir):
            index = 0
            for file in files:
                index += 1
                file_name = file
                subdir_split = subdir.split("/")
                file_full_path = ''.join([subdir,'/', file])
                file_class = subdir_split[-1]
                k_fold = index % n_folds
                if k_fold == 0:
                    k_fold = n_folds
                training_writer.writerow([file_name, file_full_path, file_class, k_fold])

# used to copy files according to each fold
def copy_images(df, directory):
    destination_directory = "./k-fold-cv/" + directory
    print("copying {} files to {}...".format(directory, destination_directory))

    # remove all files from previous fold
    if os.path.exists(destination_directory):
        shutil.rmtree(destination_directory)

    # create folder for files from this fold
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # create subfolders for each class
    for c in set(list(labels)):
        if not os.path.exists(destination_directory + '/' + c):
            os.makedirs(destination_directory + '/' + c)

    # copy files for this fold from a directory holding all the files
    for i, row in df.iterrows():
        path_to = "{}/{}".format(destination_directory, row['file_class'])
        # move files to training, test, or validation folder (the "directory" argument)
        path_from = row['file_full_path']
        # print (path_from, path_to)

        try:
            shutil.copy(path_from, path_to)
        except Exception as e:
            print("Error when copying {}: {}".format(row['file_name'], str(e)))

# https://stackoverflow.com/questions/41214527/k-fold-cross-validation-using-keras
def k_fold_cross_validation():
    model_file_suffix = date_suffix + model_version + model_extension
    # dataframe containing the filenames of the images (e.g., GUID filenames) and the classes
    df = pd.read_csv('training_file.csv')
    df_y = df['file_class']
    df_x = df['file_full_path']

    df_test = pd.read_csv('testing_file.csv')
    df_y_test = df_test['file_class']
    df_x_test = df_test['file_full_path']

    x_test, y_test = df_x_test.iloc[:], df_y_test.iloc[:]
    test = pd.concat([x_test, y_test], axis=1)
    copy_images(test, 'test')

    skf = StratifiedKFold(n_splits = n_folds)
    total_actual = []
    total_predicted = []
    total_val_accuracy = []
    total_val_loss = []
    total_test_accuracy = []

    for i, (train_index, val_index) in enumerate(skf.split(df_x, df_y)):
        x_train, x_val = df_x.iloc[train_index], df_x.iloc[val_index]
        y_train, y_val = df_y.iloc[train_index], df_y.iloc[val_index]

        train = pd.concat([x_train, y_train], axis=1)
        validation = pd.concat([x_val, y_val], axis = 1)

        # copy the images according to the fold
        copy_images(train, 'training')
        copy_images(validation, 'validation')

        print('**** Running fold '+ str(i))

        # here you call a function to create and train your model, returning validation accuracy and validation loss
        _ , res_history, model_name = transfer_learning_and_save_model_k_fold(0, inception_v3_model_name + model_file_suffix, i)
        val_accuracy = res_history.history['val_acc']
        val_loss = res_history.history['val_loss']

        # append validation accuracy and loss for average calculation later on
        total_val_accuracy.append(val_accuracy)
        total_val_loss.append(val_loss)

        # here you will call a predict() method that will predict the images on the "test" subfolder
        # this function returns the actual classes and the predicted classes in the same order
        # predicted = predict_k_fold(0, model_name)
        # ind = np.argmax(predicted,axis=1).tolist()
        # predicted = np.array([ labels[class_index] for class_index in ind]).tolist()

        actual = y_test

        # append accuracy from the predictions on the test data
        # total_test_accuracy.append(accuracy_score(actual, predicted))

        # print (type(actual))
        # print (type(total_actual))

        # append all of the actual and predicted classes for your final evaluation
        # total_actual.append(actual)
        # total_predicted.append(predicted)

        # this is optional, but you can also see the performance on each fold as the process goes on
        # print(classification_report(total_actual, total_predicted))
        # print(confusion_matrix(total_actual, total_predicted))

    print (total_val_accuracy)
    print (total_val_loss)
    print (total_test_accuracy)
    print (total_actual)
    print (total_predicted)

if __name__ == '__main__':

    model_file_suffix = date_suffix + model_version + model_extension
	
    print('[INFO] Epochs: ' + str(num_epochs))
    print('[INFO] Batch size: ' + str(batch_size))
    print('[INFO] Classes: ' + str(num_classes))
    print('[INFO] Dense layer neurons: ' + str(dense_layer_neurons))
    print('[INFO] Bruise width: ' + str(bruise_width))
    print('[INFO] Bruise height: ' + str(bruise_height))
    print('[INFO] Bruise channels: ' + str(bruise_channels))
    # print('[INFO] Learning rate: ' + str(initial_learning_rate))
    print('[INFO] Retrain layers: ' + str(split_layers_index))
    print('[INFO] Validation split: ' + str(validation_split))
    # print('[INFO] Period checkpoint: ' + str(period_checkpoint))
    print('[INFO] Training mode: ' + str(training_mode))
    print('[INFO] Resume Training mode: ' + str(resume_training_mode))
    print('[INFO] Prediction mode: ' + str(prediction_mode))
    print('[INFO] Training data: ' + str(training_data_path))
    print('[INFO] Test data: ' + str(testing_data_path))
    print('[INFO] Test image path: ' + str(image_path_test))

    if k_fold_cross_validation_mode:
        k_fold_cross_validation()

    if generate_k_fold_list_mode:
        generate_k_fold_list(training_data_path,'training_file.csv')
        # generate_k_fold_list(testing_data_path,'testing_file.csv')

    # https://www.kaggle.com/stefanie04736/simple-keras-model-with-k-fold-cross-validation
    # https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/

    if training_mode:
        transfer_learning_and_save_model(0, inception_v3_model_name + model_file_suffix)
        # transfer_learning_and_save_model(1, resnet50_model_name + model_file_suffix)
        # transfer_learning_and_save_model(2, mobilenet_model_name + model_file_suffix)

    if prediction_mode:
        print ('Using inception_v3')
        prediction = predict(0, inception_v3_model_name + model_file_suffix, image_path_test)
        print (prediction, np.argmax(prediction), labels[np.argmax(prediction)])

        print ('Using resnet50')
        prediction = predict(0, resnet50_model_name + model_file_suffix, image_path_test)
        print (prediction, np.argmax(prediction), labels[np.argmax(prediction)])

        print ('Using mobilenet')
        prediction = predict(0, mobilenet_model_name + model_file_suffix, image_path_test)
        print (prediction, np.argmax(prediction), labels[np.argmax(prediction)])

    if predict_k_fold_mode:
        predicted = predict_k_fold(0, '0_' + inception_v3_model_name + model_file_suffix)
        ind = np.argmax(predicted,axis=1).tolist()
        print (ind)
        predicted_classes = np.array([ labels[class_index] for class_index in ind]).tolist()
        print (predicted_classes)
