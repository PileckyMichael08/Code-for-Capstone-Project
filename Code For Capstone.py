# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import time

#%% Data Importing, Cleaning, and Formatting

# Import datasets and add variables for the file's source dataset and disease
# Change the sample size for each import based on your computer's computation power
Normal_TB_csv = pd.read_csv("Normal.metadata.csv")[["FILE NAME"]].sample(n=250)
Normal_TB_csv["SOURCE TYPE"] = "TB"
Normal_TB_csv["DISEASE"] = "Normal"

TB_csv = pd.read_csv("Tuberculosis.metadata.csv")[["FILE NAME"]].sample(n=500)
TB_csv["SOURCE TYPE"] = "TB"
TB_csv["DISEASE"] = "TB"


Normal_other_csv = pd.read_csv("Normal.metadata.other.csv")[["FILE NAME"]].sample(n=250)
Normal_other_csv["SOURCE TYPE"] = "Other"
Normal_other_csv["DISEASE"] = "Normal"

COVID_csv = pd.read_csv("COVID.metadata.csv")[["FILE NAME"]].sample(n=500)
COVID_csv["SOURCE TYPE"] = "Other"
COVID_csv["DISEASE"] = "Covid"

Lung_Opacity_csv = pd.read_csv("Lung_Opacity.metadata.csv")[["FILE NAME"]].sample(n=500)
Lung_Opacity_csv["SOURCE TYPE"] = "Other"
Lung_Opacity_csv["DISEASE"] = "Opacity"

Pneumonia_csv = pd.read_csv("Viral Pneumonia.metadata.csv")[["FILE NAME"]].sample(n=500)
Pneumonia_csv["SOURCE TYPE"] = "Other"
Pneumonia_csv["DISEASE"] = "Pneumonia"

# Combined Datasets
combined_df = pd.concat([Normal_TB_csv, TB_csv, Normal_other_csv, COVID_csv, Lung_Opacity_csv, Pneumonia_csv], ignore_index=True, axis=0)

# Created file path variable which returns the file path for the corresponding X-ray image
combined_df["PATH"] = ""
combined_df["PATH"].loc[(combined_df['DISEASE'] == 'Normal') & (combined_df['SOURCE TYPE'] == 'TB')] = "Normal TB/images/" + combined_df["FILE NAME"] + ".png"
combined_df["PATH"].loc[(combined_df['DISEASE'] == 'Normal') & (combined_df['SOURCE TYPE'] == 'Other')] = "Normal/images/" + combined_df["FILE NAME"] + ".png"
combined_df["PATH"].loc[(combined_df['DISEASE'] == 'TB')] = "Tuberculosis/" + combined_df["FILE NAME"] + ".png"
combined_df["PATH"].loc[(combined_df['DISEASE'] == 'Covid')] = "COVID/images/" + combined_df["FILE NAME"] + ".png"
combined_df["PATH"].loc[(combined_df['DISEASE'] == 'Opacity')] = "Lung_Opacity/images/" + combined_df["FILE NAME"] + ".png"
combined_df["PATH"].loc[(combined_df['DISEASE'] == 'Pneumonia')] = "Viral Pneumonia/images/" + combined_df["FILE NAME"] + ".png"
combined_df["PATH"]

# Uncomment the print statements below to see the number of X-ray images for each disease classification
# print(f"Number of Normal X-rays: {len(combined_df.loc[combined_df['DISEASE'] == 'Normal'])}")
# print(f"Number of X-rays with Tuberculosis: {len(combined_df.loc[combined_df['DISEASE'] == 'TB'])}")
# print(f"Number of X-rays with COVID-19: {len(combined_df.loc[combined_df['DISEASE'] == 'Covid'])}")
# print(f"Number of X-rays with Lung Opacity: {len(combined_df.loc[combined_df['DISEASE'] == 'Opacity'])}")
# print(f"Number of X-rays with Pneumonia: {len(combined_df.loc[combined_df['DISEASE'] == 'Pneumonia'])}")

# Places the file paths for 80% of the images in the training set
combined_df_train = combined_df.sample(frac = 0.8)

# Places the images not in the training set in the test set
combined_df_test = combined_df.drop(combined_df_train.index)

# Puts the training and test set file paths into their own list
path_train_list = list(combined_df_train["PATH"])

path_test_list = list(combined_df_test["PATH"])


# Places training set images in their own directory and stores them in the training set and combined dataframes
# Make sure the training directory is empty before adding images
combined_df["SPLIT_PATH"] = ""
combined_df["NEW_FILE_NAME"] = ""
combined_df_train["SPLIT_PATH"] = ""
combined_df_train["NEW_FILE_NAME"] = ""

for path in path_train_list:
    image_created = Image.open(path)
    print(path_train_list.index(path))
    if path.split('/')[0] == 'Tuberculosis':
        image_exported = image_created.save("training/" + path.split('/')[1])
        combined_df["SPLIT_PATH"].loc[(combined_df['PATH'] == path)] = "training/" + path.split('/')[1]
        combined_df["NEW_FILE_NAME"].loc[(combined_df['PATH'] == path)] = path.split('/')[1]
        combined_df_train["SPLIT_PATH"].loc[(combined_df_train['PATH'] == path)] = "training/" + path.split('/')[1]
        combined_df_train["NEW_FILE_NAME"].loc[(combined_df_train['PATH'] == path)] = path.split('/')[1]
    elif path.split('/')[0] == 'Normal TB':
        image_exported = image_created.save("training/TB_" + path.split('/')[2])
        combined_df["SPLIT_PATH"].loc[(combined_df['PATH'] == path)] = "training/TB_" + path.split('/')[2]
        combined_df["NEW_FILE_NAME"].loc[(combined_df['PATH'] == path)] = "TB_" + path.split('/')[2]
        combined_df_train["SPLIT_PATH"].loc[(combined_df_train['PATH'] == path)] = "training/TB_" + path.split('/')[2]
        combined_df_train["NEW_FILE_NAME"].loc[(combined_df_train['PATH'] == path)] = "TB_" + path.split('/')[2]
    else:
        image_exported = image_created.save("training/" + path.split('/')[2])
        combined_df["SPLIT_PATH"].loc[(combined_df['PATH'] == path)] = "training/" + path.split('/')[2]
        combined_df["NEW_FILE_NAME"].loc[(combined_df['PATH'] == path)] = path.split('/')[2]
        combined_df_train["SPLIT_PATH"].loc[(combined_df_train['PATH'] == path)] = "training/" + path.split('/')[2]
        combined_df_train["NEW_FILE_NAME"].loc[(combined_df_train['PATH'] == path)] = path.split('/')[2]

    
# Places test set images in their own directory and stores them in the test set and combined dataframes
# Make sure the test directory is empty before adding images
combined_df_test["SPLIT_PATH"] = ""
combined_df_test["NEW_FILE_NAME"] = ""

for path in path_test_list:
    image_created = Image.open(path)
    print(path_test_list.index(path))
    if path.split('/')[0] == 'Tuberculosis':
        image_exported = image_created.save("testing/" + path.split('/')[1])
        combined_df["SPLIT_PATH"].loc[(combined_df['PATH'] == path)] = "testing/" + path.split('/')[1]
        combined_df["NEW_FILE_NAME"].loc[(combined_df['PATH'] == path)] = path.split('/')[1]
        combined_df_test["SPLIT_PATH"].loc[(combined_df_test['PATH'] == path)] = "testing/" + path.split('/')[1]
        combined_df_test["NEW_FILE_NAME"].loc[(combined_df_test['PATH'] == path)] = path.split('/')[1]
        
    elif path.split('/')[0] == 'Normal TB':
        image_exported = image_created.save("testing/TB_" + path.split('/')[2])
        combined_df["SPLIT_PATH"].loc[(combined_df['PATH'] == path)] = "testing/TB_" + path.split('/')[2]
        combined_df["NEW_FILE_NAME"].loc[(combined_df['PATH'] == path)] = "TB_" + path.split('/')[2]
        combined_df_test["SPLIT_PATH"].loc[(combined_df_test['PATH'] == path)] = "testing/TB_" + path.split('/')[2]
        combined_df_test["NEW_FILE_NAME"].loc[(combined_df_test['PATH'] == path)] = "TB_" + path.split('/')[2]
    else:
        image_exported = image_created.save("testing/" + path.split('/')[2])
        combined_df["SPLIT_PATH"].loc[(combined_df['PATH'] == path)] = "testing/" + path.split('/')[2]
        combined_df["NEW_FILE_NAME"].loc[(combined_df['PATH'] == path)] = path.split('/')[2]
        combined_df_test["SPLIT_PATH"].loc[(combined_df_test['PATH'] == path)] = "testing/" + path.split('/')[2]
        combined_df_test["NEW_FILE_NAME"].loc[(combined_df_test['PATH'] == path)] = path.split('/')[2]


#%% ResNet50

# Creates an image data generator for the ResNet50 model
image_generator_resnet = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization= True,
    zoom_range = 0.1,
    rotation_range = 5,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

# Stores the X-ray images stored in the training directory and their disease classifications into
# a ResNet50 image data generator
train_generator_resnet = image_generator_resnet.flow_from_dataframe( dataframe=combined_df_train,
                                                           directory='training',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))


# Stores the X-ray images stored in the test directory and their disease classifications into
# a ResNet50 image data generator
test_generator_resnet = image_generator_resnet.flow_from_dataframe( dataframe=combined_df_test,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))

# Creates the base ResNet50 model
model_resnet = tf.keras.applications.resnet50.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg')

# Adds dropout, flatten, and dense layers to the ResNet50 model
model_resnet.trainable = False
resnet_inputs = model_resnet.input
resnet_dropout = tf.keras.layers.Dropout(.1)(model_resnet.output)
resnet_flatten = tf.keras.layers.Flatten()(resnet_dropout)
resnet_d1 = tf.keras.layers.Dense(1024, activation='relu')(resnet_flatten)
resnet_d2 = tf.keras.layers.Dense(512, activation='relu')(resnet_d1)
resnet_outputs = tf.keras.layers.Dense(5, activation='softmax')(resnet_d2 )
resnet_model = tf.keras.Model(resnet_inputs, resnet_outputs)
resnet_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# A callback function that stops model training after validation loss stops improving
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)


# Fits the ResNet50 model to the image data generators
resnet_history = resnet_model.fit(train_generator_resnet,
                              steps_per_epoch=10,
                              epochs=5,
                              validation_data = test_generator_resnet,
                              validation_steps = 5,
                              callbacks=[callbacks])


# start, end, and diff compute how long it takes for the ResNet50 model to evaluate the test set images
start = time.time()

# Evaluates the fitted ResNet50 model
resnet_scores = resnet_model.evaluate(test_generator_resnet)

end = time.time()

diff = end - start

print(f"Time it took to run ResNet Model is {diff} seconds")


# Generates a confusion matrix for ResNet50 model predictions
resnet_predictions = np.argmax(resnet_model.predict(test_generator_resnet), axis=1)
resnet_matrix = confusion_matrix(test_generator_resnet.labels, resnet_predictions)

labels = ['Covid','Normal','Opacity','Pneumonia','TB']
disp = ConfusionMatrixDisplay(confusion_matrix=resnet_matrix, display_labels= labels)
fig, ax = plt.subplots(figsize=(6,6)) 
disp.plot(ax=ax) 
plt.title("ResNet50 Confusion Matrix")
plt.xlabel("Predicted", fontsize = 15)
plt.ylabel("Actual", fontsize = 15)
plt.show()
fig.savefig("ResNet50 Confusion Matrix",dpi=700)

# Displays the number of images that fall into each disease classification including the data source of the normal images
normal_source_test_df = pd.DataFrame()
normal_source_test_df["DISEASE_NORMAL_DIFF"] = combined_df_test["DISEASE_NORMAL_DIFF"]
normal_source_test_df["ResNet Predictions"] = resnet_predictions
print(normal_source_test_df["DISEASE_NORMAL_DIFF"].value_counts())

# Displays the number of times the ResNet50 model predicted each disease classification for the normal images from each data source 
print(normal_source_test_df.loc[(normal_source_test_df["DISEASE_NORMAL_DIFF"] == 'Normal TB')]["ResNet Predictions"].value_counts())
print(normal_source_test_df.loc[(normal_source_test_df["DISEASE_NORMAL_DIFF"] == 'Normal Other')]["ResNet Predictions"].value_counts())




# Creates dataframes for each disease classification including normal images for each data source
combined_df_test_covid = combined_df_test.loc[(combined_df_test["DISEASE"] == 'Covid')]

combined_df_test_normal = combined_df_test.loc[(combined_df_test["DISEASE"] == 'Normal')]

combined_df_test_opacity = combined_df_test.loc[(combined_df_test["DISEASE"] == 'Opacity')]

combined_df_test_pneumonia = combined_df_test.loc[(combined_df_test["DISEASE"] == 'Pneumonia')]

combined_df_test_tb = combined_df_test.loc[(combined_df_test["DISEASE"] == 'TB')]

combined_df_test_tb_source = combined_df_test.loc[(combined_df_test['DISEASE_NORMAL_DIFF'] == 'Normal TB')]

combined_df_test_other_source = combined_df_test.loc[(combined_df_test['DISEASE_NORMAL_DIFF'] == 'Normal Other')]

# Creates image generators for each disease classification specific dataframe
test_generator_resnet_covid = image_generator_resnet.flow_from_dataframe( dataframe=combined_df_test_covid,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))

test_generator_resnet_normal = image_generator_resnet.flow_from_dataframe( dataframe=combined_df_test_normal,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))

test_generator_resnet_opacity = image_generator_resnet.flow_from_dataframe( dataframe=combined_df_test_opacity,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))

test_generator_resnet_pneumonia = image_generator_resnet.flow_from_dataframe( dataframe=combined_df_test_pneumonia,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))

test_generator_resnet_tb = image_generator_resnet.flow_from_dataframe( dataframe=combined_df_test_tb,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))

test_generator_resnet_tb_source = image_generator_resnet.flow_from_dataframe( dataframe=combined_df_test_tb_source,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))

test_generator_resnet_other_source = image_generator_resnet.flow_from_dataframe( dataframe=combined_df_test_other_source,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))


# Returns the amount of milliseconds it for the ResNet50 model to evaluate each image image on average
# from each disease classification specific image generator
start = time.time()
resnet_scores_covid = resnet_model.evaluate(test_generator_resnet_covid)

end = time.time()

diff = end - start

print(f"Time it took to run Covid Images in ResNet Model is {round(((diff / 101) * 1000),2)} milliseconds")

start = time.time()
resnet_scores_normal = resnet_model.evaluate(test_generator_resnet_normal)

end = time.time()

diff = end - start

print(f"Time it took to run Normal Images in ResNet Model is {round(((diff / 105) * 1000),2)} milliseconds")

start = time.time()
resnet_scores_opacity = resnet_model.evaluate(test_generator_resnet_opacity)

end = time.time()

diff = end - start

print(f"Time it took to run Opacity Images in ResNet Model is {round(((diff / 93) * 1000),2)} milliseconds")

start = time.time()
resnet_scores_pneumonia = resnet_model.evaluate(test_generator_resnet_pneumonia)

end = time.time()

diff = end - start

print(f"Time it took to run Pneumonia Images in ResNet Model is {round(((diff / 99) * 1000),2)} milliseconds")

start = time.time()
resnet_scores_tb = resnet_model.evaluate(test_generator_resnet_tb)

end = time.time()

diff = end - start

print(f"Time it took to run TB Images in ResNet Model is {round(((diff / 102) * 1000),2)} milliseconds")

start = time.time()
resnet_scores_tb_source = resnet_model.evaluate(test_generator_resnet_tb_source)

end = time.time()

diff = end - start

print(f"Time it took to run TB Source Images in ResNet Model is {round(((diff / 53) * 1000),2)} milliseconds")

start = time.time()
resnet_scores_other_source = resnet_model.evaluate(test_generator_resnet_other_source)

end = time.time()

diff = end - start

print(f"Time it took to run Other Source Images in ResNet Model is {round(((diff / 52) * 1000),2)} milliseconds")

#%% DenseNet201

# Creates an image data generator for the DenseNet201 model
image_generator_densenet = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization= True,
    zoom_range = 0.1,
    rotation_range = 5,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    preprocessing_function=tf.keras.applications.densenet.preprocess_input)

# Stores the X-ray images stored in the training directory and their disease classifications into
# a DenseNet201 image data generator
train_generator_densenet = image_generator_densenet.flow_from_dataframe( dataframe=combined_df_train,
                                                           directory='training',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))


# Stores the X-ray images stored in the test directory and their disease classifications into
# a DenseNet201 image data generator
test_generator_densenet = image_generator_densenet.flow_from_dataframe( dataframe=combined_df_test,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))

# Creates the base DenseNet201 model
model_densenet = tf.keras.applications.densenet.DenseNet201(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg')

# Adds dropout, flatten, and dense layers to the DenseNet201 model
model_densenet.trainable = False
densenet_inputs = model_densenet.input
densenet_dropout = tf.keras.layers.Dropout(.1)(model_densenet.output)
densenet_flatten = tf.keras.layers.Flatten()(densenet_dropout)
densenet_d1 = tf.keras.layers.Dense(1024, activation='relu')(densenet_flatten)
densenet_d2 = tf.keras.layers.Dense(512, activation='relu')(densenet_d1)
densenet_outputs = tf.keras.layers.Dense(5, activation='softmax')(densenet_d2 )
densenet_model = tf.keras.Model(densenet_inputs, densenet_outputs)
densenet_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# A callback function that stops model training after validation loss stops improving
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)

# Fits the DenseNet201 model to the image data generators
densenet_history = densenet_model.fit(train_generator_densenet,
                              steps_per_epoch=10,
                              epochs=5,
                              validation_data = test_generator_densenet,
                              validation_steps = 5,
                              callbacks=[callbacks])

# start, end, and diff compute how long it takes for the DenseNet201 model to evaluate the test set images
start = time.time()

# Evaluates the fitted DenseNet201 model
densenet_scores = densenet_model.evaluate(test_generator_densenet)

end = time.time()

diff = end - start

print(f"Time it took to run DenseNet Model is {diff} seconds")


# Generates a confusion matrix for DenseNet201 model predictions
densenet_predictions = np.argmax(densenet_model.predict(test_generator_densenet), axis=1)
densenet_matrix = confusion_matrix(test_generator_densenet.labels, densenet_predictions)
densenet_report = classification_report(test_generator_densenet.labels, densenet_predictions, 
                                      target_names=test_generator_densenet.class_indices, zero_division=0)

labels = ['Covid','Normal','Opacity','Pneumonia','TB']
disp = ConfusionMatrixDisplay(confusion_matrix=densenet_matrix, display_labels= labels)
fig, ax = plt.subplots(figsize=(6,6)) 
disp.plot(ax=ax) 
plt.title("DenseNet201 Confusion Matrix")
plt.xlabel("Predicted", fontsize = 15)
plt.ylabel("Actual", fontsize = 15)
plt.show()
fig.savefig("DenseNet201 Confusion Matrix",dpi=700)

# Displays the number of images that fall into each disease classification including the data source of the normal images
normal_source_test_df = pd.DataFrame()
normal_source_test_df["DISEASE_NORMAL_DIFF"] = combined_df_test["DISEASE_NORMAL_DIFF"]
normal_source_test_df["DenseNet Predictions"] = densenet_predictions
print(normal_source_test_df["DISEASE_NORMAL_DIFF"].value_counts())

# Displays the number of times the DenseNet201 model predicted each disease classification for the normal images from each data source 
print(normal_source_test_df.loc[(normal_source_test_df["DISEASE_NORMAL_DIFF"] == 'Normal TB')]["DenseNet Predictions"].value_counts())
print(normal_source_test_df.loc[(normal_source_test_df["DISEASE_NORMAL_DIFF"] == 'Normal Other')]["DenseNet Predictions"].value_counts())

# Creates image generators for each disease classification specific dataframe
# The disease classification specific dataframes can be found in the ResNet50 section
test_generator_densenet_covid = image_generator_densenet.flow_from_dataframe( dataframe=combined_df_test_covid,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))

test_generator_densenet_normal = image_generator_densenet.flow_from_dataframe( dataframe=combined_df_test_normal,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))

test_generator_densenet_opacity = image_generator_densenet.flow_from_dataframe( dataframe=combined_df_test_opacity,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))

test_generator_densenet_pneumonia = image_generator_densenet.flow_from_dataframe( dataframe=combined_df_test_pneumonia,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))

test_generator_densenet_tb = image_generator_densenet.flow_from_dataframe( dataframe=combined_df_test_tb,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))

test_generator_densenet_tb_source = image_generator_densenet.flow_from_dataframe( dataframe=combined_df_test_tb_source,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))

test_generator_densenet_other_source = image_generator_densenet.flow_from_dataframe( dataframe=combined_df_test_other_source,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(224,224))


# Returns the amount of milliseconds it for the DenseNet201 model to evaluate each image image on average
# from each disease classification specific image generator
start = time.time()
densenet_scores_covid = densenet_model.evaluate(test_generator_densenet_covid)

end = time.time()

diff = end - start

print(f"Time it took to run Covid Images in DenseNet Model is {round(((diff / 101) * 1000),2)} milliseconds")

start = time.time()
densenet_scores_normal = densenet_model.evaluate(test_generator_densenet_normal)

end = time.time()

diff = end - start

print(f"Time it took to run Normal Images in DenseNet Model is {round(((diff / 105) * 1000),2)} milliseconds")

start = time.time()
densenet_scores_opacity = densenet_model.evaluate(test_generator_densenet_opacity)

end = time.time()

diff = end - start

print(f"Time it took to run Opacity Images in DenseNet Model is {round(((diff / 93) * 1000),2)} milliseconds")

start = time.time()
densenet_scores_pneumonia = densenet_model.evaluate(test_generator_densenet_pneumonia)

end = time.time()

diff = end - start

print(f"Time it took to run Pneumonia Images in DenseNet Model is {round(((diff / 99) * 1000),2)} milliseconds")

start = time.time()
densenet_scores_tb = densenet_model.evaluate(test_generator_densenet_tb)

end = time.time()

diff = end - start

print(f"Time it took to run TB Images in DenseNet Model is {round(((diff / 102) * 1000),2)} milliseconds")

start = time.time()
densenet_scores_tb_source = densenet_model.evaluate(test_generator_densenet_tb_source)

end = time.time()

diff = end - start

print(f"Time it took to run TB Source Images in DenseNet Model is {round(((diff / 53) * 1000),2)} milliseconds")

start = time.time()
densenet_scores_other_source = densenet_model.evaluate(test_generator_densenet_other_source)

end = time.time()

diff = end - start

print(f"Time it took to run Other Source Images in DenseNet Model is {round(((diff / 52) * 1000),2)} milliseconds")





#%% InceptionV3

# Creates an image data generator for the InceptionV3 model
image_generator_inceptionv3 = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization= True,
    zoom_range = 0.1,
    rotation_range = 5,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    preprocessing_function=tf.keras.applications.inception_v3.preprocess_input)

# Stores the X-ray images stored in the training directory and their disease classifications into
# a InceptionV3 image data generator
train_generator_inceptionv3 = image_generator_inceptionv3.flow_from_dataframe( dataframe=combined_df_train,
                                                           directory='training',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(299,299))


# Stores the X-ray images stored in the test directory and their disease classifications into
# a InceptionV3 image data generator
test_generator_inceptionv3 = image_generator_inceptionv3.flow_from_dataframe( dataframe=combined_df_test,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(299,299))

# Creates the base InceptionV3 model
model_inceptionv3 = tf.keras.applications.inception_v3.InceptionV3(
    input_shape=(299, 299, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg')

# Adds dropout, flatten, and dense layers to the InceptionV3 model
model_inceptionv3.trainable = False
inceptionv3_inputs = model_inceptionv3.input
inceptionv3_dropout = tf.keras.layers.Dropout(.1)(model_inceptionv3.output)
inceptionv3_flatten = tf.keras.layers.Flatten()(inceptionv3_dropout)
inceptionv3_d1 = tf.keras.layers.Dense(1024, activation='relu')(inceptionv3_flatten)
inceptionv3_d2 = tf.keras.layers.Dense(512, activation='relu')(inceptionv3_d1)
inceptionv3_outputs = tf.keras.layers.Dense(5, activation='softmax')(inceptionv3_d2)
inceptionv3_model = tf.keras.Model(inceptionv3_inputs, inceptionv3_outputs)
inceptionv3_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# A callback function that stops model training after validation loss stops improving
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)

# Fits the InceptionV3 model to the image data generators
inceptionv3_history = inceptionv3_model.fit(train_generator_inceptionv3,
                              steps_per_epoch=10,
                              epochs=5,
                              validation_data = test_generator_inceptionv3,
                              validation_steps = 5,
                              callbacks=[callbacks]) 


# start, end, and diff compute how long it takes for the InceptionV3 model to evaluate the test set images
start = time.time()

# Evaluates the fitted InceptionV3 model
inceptionv3_scores = inceptionv3_model.evaluate(test_generator_inceptionv3)

end = time.time()

diff = end - start

print(f"Time it took to run Inception Model is {diff} seconds")


# Generates a confusion matrix for InceptionV3 model predictions
inceptionv3_predictions = np.argmax(inceptionv3_model.predict(test_generator_inceptionv3), axis=1)
inceptionv3_matrix = confusion_matrix(test_generator_inceptionv3.labels, inceptionv3_predictions)
inceptionv3_report = classification_report(test_generator_inceptionv3.labels, inceptionv3_predictions, 
                                      target_names=test_generator_inceptionv3.class_indices, zero_division=0)


labels = ['Covid','Normal','Opacity','Pneumonia','TB']
disp = ConfusionMatrixDisplay(confusion_matrix=inceptionv3_matrix, display_labels= labels)
fig, ax = plt.subplots(figsize=(6,6)) 
disp.plot(ax=ax) 
plt.title("Inception V3 Confusion Matrix")
plt.xlabel("Predicted", fontsize = 15)
plt.ylabel("Actual", fontsize = 15)
plt.show()
fig.savefig("Inception V3 Confusion Matrix",dpi=700)

# Displays the number of images that fall into each disease classification including the data source of the normal images
normal_source_test_df = pd.DataFrame()
normal_source_test_df["DISEASE_NORMAL_DIFF"] = combined_df_test["DISEASE_NORMAL_DIFF"]
normal_source_test_df["Inception Predictions"] = inceptionv3_predictions
print(normal_source_test_df["DISEASE_NORMAL_DIFF"].value_counts())

# Displays the number of times the InceptionV3 model predicted each disease classification for the normal images from each data source 
print(normal_source_test_df.loc[(normal_source_test_df["DISEASE_NORMAL_DIFF"] == 'Normal TB')]["Inception Predictions"].value_counts())
print(normal_source_test_df.loc[(normal_source_test_df["DISEASE_NORMAL_DIFF"] == 'Normal Other')]["Inception Predictions"].value_counts())


# Creates image generators for each disease classification specific dataframe
# The disease classification specific dataframes can be found in the ResNet50 section
test_generator_inceptionv3_covid = image_generator_inceptionv3.flow_from_dataframe( dataframe=combined_df_test_covid,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(299,299))

test_generator_inceptionv3_normal = image_generator_inceptionv3.flow_from_dataframe( dataframe=combined_df_test_normal,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(299,299))

test_generator_inceptionv3_opacity = image_generator_inceptionv3.flow_from_dataframe( dataframe=combined_df_test_opacity,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(299,299))

test_generator_inceptionv3_pneumonia = image_generator_inceptionv3.flow_from_dataframe( dataframe=combined_df_test_pneumonia,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(299,299))

test_generator_inceptionv3_tb = image_generator_inceptionv3.flow_from_dataframe( dataframe=combined_df_test_tb,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(299,299))

test_generator_inceptionv3_tb_source = image_generator_inceptionv3.flow_from_dataframe( dataframe=combined_df_test_tb_source,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(299,299))

test_generator_inceptionv3_other_source = image_generator_inceptionv3.flow_from_dataframe( dataframe=combined_df_test_other_source,
                                                           directory='testing',
                                                           x_col='NEW_FILE_NAME',
                                                           y_col='DISEASE',
                                                           class_mode="categorical",
                                                           batch_size=8,
                                                           shuffle=True,
                                                           seed=8,
                                                           target_size=(299,299))


# Returns the amount of milliseconds it for the InceptionV3 model to evaluate each image image on average
# from each disease classification specific image generator
start = time.time()
inceptionv3_scores_covid = inceptionv3_model.evaluate(test_generator_inceptionv3_covid)

end = time.time()

diff = end - start

print(f"Time it took to run Covid Images in InceptionV3 Model is {round(((diff / 101) * 1000),2)} milliseconds")

start = time.time()
inceptionv3_scores_normal = inceptionv3_model.evaluate(test_generator_inceptionv3_normal)

end = time.time()

diff = end - start

print(f"Time it took to run Normal Images in InceptionV3 Model is {round(((diff / 105) * 1000),2)} milliseconds")

start = time.time()
inceptionv3_scores_opacity = inceptionv3_model.evaluate(test_generator_inceptionv3_opacity)

end = time.time()

diff = end - start

print(f"Time it took to run Opacity Images in InceptionV3 Model is {round(((diff / 93) * 1000),2)} milliseconds")

start = time.time()
inceptionv3_scores_pneumonia = inceptionv3_model.evaluate(test_generator_inceptionv3_pneumonia)

end = time.time()

diff = end - start

print(f"Time it took to run Pneumonia Images in InceptionV3 Model is {round(((diff / 99) * 1000),2)} milliseconds")

start = time.time()
inceptionv3_scores_tb = inceptionv3_model.evaluate(test_generator_inceptionv3_tb)

end = time.time()

diff = end - start

print(f"Time it took to run TB Images in InceptionV3 Model is {round(((diff / 102) * 1000),2)} milliseconds")

start = time.time()
inceptionv3_scores_tb_source = inceptionv3_model.evaluate(test_generator_inceptionv3_tb_source)

end = time.time()

diff = end - start

print(f"Time it took to run TB Source Images in InceptionV3 Model is {round(((diff / 53) * 1000),2)} milliseconds")

start = time.time()
inceptionv3_scores_other_source = inceptionv3_model.evaluate(test_generator_inceptionv3_other_source)

end = time.time()

diff = end - start

print(f"Time it took to run Other Source Images in InceptionV3 Model is {round(((diff / 52) * 1000),2)} milliseconds")



# %% Figures
# %%% Figure 1

fig_1_model_types = ["InceptionV3", "DenseNet201", "ResNet50"]
# Accuracies can be found when fitting each model
fig_1_training_accuracies = [62.23, 62.00, 57.39]
fig_1_test_accuracies = [50.00, 55.00, 47.5]

# Sets up the x-axis labels and bars
fig_1_x_axis = np.arange(len(fig_1_model_types))   
bar_1 = plt.bar(fig_1_x_axis - 0.2, fig_1_training_accuracies, 0.4, label = 'Training') 
bar_2 = plt.bar(fig_1_x_axis + 0.2, fig_1_test_accuracies, 0.4, label = 'Test') 

# Creates graph and graph labels
plt.xticks(fig_1_x_axis, fig_1_model_types) 
plt.xlabel("CNN Model Types") 
plt.ylabel("Disease Classification Accuracy") 
#plt.title("DenseNet201 and InceptionV3 Models\nHave the Best Classification Accuracy")
plt.legend(title = "Data Type", loc='center right', bbox_to_anchor=(1.25, 0.5))

# Generates the percents shown above each bar
for rect in bar_1 + bar_2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}%', ha='center', va='bottom')

# Sets y-axis format to percent format
vals = ax.get_yticks()
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.show() 

# %%% Figure 2

fig_2_model_types = ["ResNet50", "DenseNet201", "InceptionV3"]
# Outputs can be found by finding the percent difference between the Normal TB accuracy and Normal Other accuacy
fig_2_accuracies = [65.03, 65.00, 8.68]

# Sets up the x-axis labels and bars
fig_2_x_axis = np.arange(len(fig_2_model_types)) 
bar_1 = plt.bar(fig_2_x_axis, fig_2_accuracies) 

# Creates graph and graph labels
plt.xticks(fig_2_x_axis, fig_2_model_types) 
plt.xlabel("CNN Model Types") 
plt.ylabel("Percent Difference in Normal Image Classification") 
#plt.title("InceptionV3 Model Handles\nImages from Multiple Sources the Best")

# Generates the percents shown above each bar
for rect in bar_1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}%', ha='center', va='bottom')

# Sets y-axis format to percent format
vals = ax.get_yticks()
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.show()

# %%% Figure 3

fig_3_model_types = ["DenseNet201", "InceptionV3", "ResNet50"]
# Outputs can be found taking time generated to run the model evaluation divided by the number of images
# in the test set. Remember that the time is in milliseconds
fig_3_accuracies = [389, 286, 226]

# Sets up the x-axis labels and bars
fig_3_x_axis = np.arange(len(fig_3_model_types))  
bar_1 = plt.bar(fig_3_x_axis, fig_3_accuracies) 

# Creates graph and graph labels
plt.xticks(fig_3_x_axis, fig_3_model_types) 
plt.xlabel("CNN Model Types") 
plt.ylabel("Average Time Needed to Evaluate Each Image\n(Milliseconds)") 
#plt.title("ResNet50 Model Evaluates Each X-ray Image the Fastest on Average")

# Generates the numbers shown above each bar
for rect in bar_1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')
    
plt.show()

# %%% Figure 4

fig_4_model_types = ["ResNet50", "DenseNet201", "InceptionV3"]
fig_4_disease_cats = ["Covid-19", "Normal", "Opacity", "Pneumonia", "TB"]

# Accuracies can be found by taking the number found where the model confusion matrix overlaps and
# divide it by the total number of images in the test set
fig_4_resnet_accuracies = [25.7, 9.5, 13.98, 40.4, 24.5]
fig_4_desnet_accuracies = [1.0, 8.6, 32.3, 22.2, 35.3]
fig_4_inception_accuracies = [1.0, 25.7, 34.4, 19.2, 24.5]

# Sets up the x-axis labels and bars
fig_4_x_axis = np.arange(len(fig_4_disease_cats))   
bar_1 = plt.bar(fig_4_x_axis - 0.2, fig_4_resnet_accuracies, 0.2, label = 'ResNet50') 
bar_2 = plt.bar(fig_4_x_axis, fig_4_desnet_accuracies, 0.2, label = 'DenseNet201')
bar_3 = plt.bar(fig_4_x_axis + 0.2, fig_4_inception_accuracies, 0.2, label = 'InceptionV3') 
 
# Creates graph and graph labels 
plt.xticks(fig_4_x_axis, fig_4_disease_cats) 
plt.xlabel("Disease Classifications") 
plt.ylabel("Disease Classification Accuracy") 
#plt.title("ResNet50 Model Has the Most\nBalanced Classification Accuracy")
plt.legend(title = "Model Type", loc='center right', bbox_to_anchor=(1.3, 0.5))

# Sets y-axis format to percent format
vals = ax.get_yticks()
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.show() 

# %%% Figure 5

fig_5_model_types = ["ResNet50", "DenseNet201", "InceptionV3"]
# Outputs can be found by finding the average ranking each model's classification accuracy for each disease category
# Ranks go 1, 2, then 3. Ties result in a half being added to the rank. So, a tie for second is 2.5
fig_5_placements = [1.9, 2, 2.1]

# Sets up the x-axis labels and bars
fig_5_x_axis = np.arange(len(fig_5_placements)) 
bar_1 = plt.bar(fig_5_x_axis, fig_5_placements) 

# Creates graph and graph labels  
plt.xticks(fig_5_x_axis, fig_5_model_types) 
plt.xlabel("CNN Model Types") 
plt.ylabel("Average Disease Classification Accuracy Placement") 
#plt.title("ResNet50 Model Barely Has the Best\nAverage Disease Classification Rank")

# Generates the numbers shown above each bar
for rect in bar_1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.1f}', ha='center', va='bottom')
    
plt.show()

# %%% Figure 6

fig_6_model_types = ["ResNet50", "DenseNet201", "InceptionV3"]
# TB accuracies can be found by taking the number found where the TB sections of the model confusion matrix overlaps and
# add it to the number of Normal TB images that were predicted correctly. Then divide this number by the total number 
#of TB and Normal TB images in the test set
fig_6_TB_accuracies = [17.42, 27.10, 25.81]
# The other accuracies can be found following the same process as the TB accuracies for the other classifications
fig_6_other_accuracies = [25.21, 16.23, 18.55]

# Sets up the x-axis labels and bars
fig_6_x_axis = np.arange(len(fig_6_model_types))  
bar_1 = plt.bar(fig_6_x_axis - 0.2, fig_6_TB_accuracies, 0.4, label = 'TB') 
bar_2 = plt.bar(fig_6_x_axis + 0.2, fig_6_other_accuracies, 0.4, label = 'Other Diseases') 

# Creates graph and graph labels  
plt.xticks(fig_6_x_axis, fig_6_model_types) 
plt.xlabel("CNN Model Types") 
plt.ylabel("Disease Classification Accuracy") 
#plt.title("InceptionV3 Model Has the Smallest Difference\nin Classification Accuracy Between Sources")
plt.legend(title = "Data Source Type", loc='center right', bbox_to_anchor=(1.31, 0.5))

# Generates the numbers shown above each bar
for rect in bar_1 + bar_2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}%', ha='center', va='bottom')

# Sets y-axis format to percent format
vals = ax.get_yticks()
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.show() 

# %%% Figure 7

fig_7_model_types = ["ResNet50", "DenseNet201", "InceptionV3"]

# Outputs can be found by finding the percent difference between the accuracy found in Figure 4 by TB accuracies
# found in Figure 6. Compare the TB accuracies to the Normal Other accuracies for the normal accuracy section
fig_7_covid_accuracy_diffs = [38.4, 185.77, 185.08]
fig_7_normal_accuracy_diffs = [12.44, 129.78, 11.17]
fig_7_opacity_accuracy_diffs = [21.91, 17.51, 28.53]
fig_7_pneumonia_accuracy_diffs = [79.49, 19.88, 29.37]

# Sets up the x-axis labels and bars
fig_7_x_axis = np.arange(len(fig_7_model_types)) 
bar_1 = plt.bar(fig_7_x_axis - 0.3, fig_7_covid_accuracy_diffs, 0.15, label = 'Covid-19') 
bar_2 = plt.bar(fig_7_x_axis - 0.15, fig_7_normal_accuracy_diffs, 0.15, label = 'Normal')   
bar_3 = plt.bar(fig_7_x_axis, fig_7_opacity_accuracy_diffs, 0.15, label = 'Opacity')
bar_4 = plt.bar(fig_7_x_axis + 0.15, fig_7_pneumonia_accuracy_diffs, 0.15, label = 'Pneumonia') 

# Creates graph and graph labels 
plt.xticks(fig_7_x_axis, fig_7_model_types) 
plt.xlabel("CNN Model Types") 
plt.ylabel("Disease Classification Accuracy\nPercent Difference From TB Source") 
#plt.title("ResNet Model Has the Smallest Accuracy Difference\nBetween TB Source and Other Diseases")
plt.legend(title = "Diseases Compared to\nTB Source", loc='center right', bbox_to_anchor=(1.31, 0.5))

# Sets y-axis format to percent format
vals = ax.get_yticks()
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.show() 

# %%% Figure 8

fig_8_disease_cats = ["Covid-19", "Normal", "Opacity", "Pneumonia", "TB"]

# Times can be found by following the time code at the end of each model section
fig_8_resnet_times = [25.61, 26.28, 25.69, 25.58, 26.12]
fig_8_desnet_times = [106.91, 46.41, 46.24, 45.69, 46.97]
fig_8_inception_times = [52.06, 30.21, 31.16, 31.77, 34.00]

# Sets up the x-axis labels and bars
fig_8_x_axis = np.arange(len(fig_8_disease_cats)) 
bar_1 = plt.bar(fig_8_x_axis - 0.2, fig_8_resnet_times, 0.2, label = 'ResNet50') 
bar_2 = plt.bar(fig_8_x_axis, fig_8_desnet_times, 0.2, label = 'DenseNet201')
bar_3 = plt.bar(fig_8_x_axis + 0.2, fig_8_inception_times, 0.2, label = 'InceptionV3') 
 
# Creates graph and graph labels  
plt.xticks(fig_8_x_axis, fig_8_disease_cats) 
plt.xlabel("Disease Classifications") 
plt.ylabel("Average Time Needed to Evaluate Each Image\n(Milliseconds)") 
#plt.title("ResNet50 Model is the Fastest for Each\nDisease and is the Most Balanced")
plt.legend(title = "Model Type", loc='center right', bbox_to_anchor=(1.3, 0.5))
plt.show() 

# %%% Figure 9

fig_9_model_types = ["ResNet50", "DenseNet201", "InceptionV3"]

# Times can be found by following the time code at the end of each model section
fig_9_TB_accuracies = [27.08, 52.12, 35.5]
fig_9_other_accuracies = [27.98, 52.52, 33.91]

# Sets up the x-axis labels and bars
fig_9_x_axis = np.arange(len(fig_9_model_types)) 
bar_1 = plt.bar(fig_9_x_axis - 0.2, fig_9_TB_accuracies, 0.4, label = 'TB') 
bar_2 = plt.bar(fig_9_x_axis + 0.2, fig_9_other_accuracies, 0.4, label = 'Other Diseases') 

# Creates graph and graph labels 
plt.xticks(fig_9_x_axis, fig_9_model_types) 
plt.xlabel("CNN Model Types") 
plt.ylabel("Average Time Needed to Evaluate Each Normal\nImage (Milliseconds)")
#plt.title("InceptionV3 Model Computation Speed is the Most Affected by Using Different Data Sources")
plt.legend(title = "Data Source Type", loc='center right', bbox_to_anchor=(1.31, 0.5))

# Generates the numbers shown above each bar
for rect in bar_1 + bar_2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')

plt.show() 











































