# --------------------------------------------------
# Standard Imports
# --------------------------------------------------
import os
import random
from copy import deepcopy
import datetime
from sklearn.utils import shuffle
from collections import defaultdict

# --------------------------------------------------
# Data Manipulation
# --------------------------------------------------
import numpy as np
import pandas as pd

# --------------------------------------------------
# Image Processing
# --------------------------------------------------
from PIL import Image, ImageEnhance, ImageOps
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --------------------------------------------------
# Machine Learning & Deep Learning
# --------------------------------------------------
import shutil
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, precision_recall_curve, auc)

# Resampling
from imblearn.over_sampling import RandomOverSampler

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras import backend as K, layers, models
from tensorflow.keras.models import Model, Sequential, load_model
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Input, Concatenate
from tensorflow.keras.layers import BatchNormalization, LeakyReLU 
from tensorflow.keras.regularizers import l2 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.applications import VGG16, ResNet50

# --------------------------------------------------
# Visualization
# --------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots

# Evaluation Metrics for Visualization
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve

# -------------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------
# Visualization Functions
# --------------------------------------------------

def bar_plot(data, column_name, title, x_label, figsize, hue = None, show_legend = False, y_label = 'Number of observations'):
    """
    Creates a bar plot to visualize the distribution of data in a specified column.
    
    Parameters:
        data (pd.DataFrame): The data containing the column to be visualized.
        column_name (str): The column name to visualize.
        title (str): Title of the plot.
        x_label (str): Label for the x-axis.
        figsize (tuple): Size of the figure (width, height).
        hue (str, optional): Column name to color the bars by. Defaults to None.
        show_legend (bool, optional): Whether to show the legend. Defaults to False.
        y_label (str, optional): Label for the y-axis. Defaults to 'Number of observations'.
        
    Returns:
        None
    """
    plt.figure(figsize = figsize)
    sns.countplot(data = data, x = column_name, palette = 'flare', order = data[column_name].value_counts().index, hue = hue)
    plt.xlabel(x_label, fontsize = 12)
    plt.ylabel(y_label, fontsize = 12)
    plt.title(title, fontsize = 16)
    
    if show_legend:
        plt.legend(title = hue)
    else:
        plt.legend().remove()
    plt.show()

def heatmap(data, index, columns, aggfunc, fill_value, figsize, title):
    """
    Creates a heatmap to visualize the aggregated data in a pivot table format.
    
    Parameters:
        data (pd.DataFrame): The data to be pivoted and visualized.
        index (str): Column name to be used as index.
        columns (str): Column name to be used as columns.
        aggfunc (str or function): Aggregation function to compute on the pivot table.
        fill_value (scalar): Value to replace missing values in the pivot table.
        figsize (tuple): Size of the figure (width, height).
        title (str): Title of the plot.
        
    Returns:
        None
    """
    pivot_table = data.pivot_table(index = index, columns = columns, aggfunc = aggfunc, fill_value = fill_value)
    plt.figure(figsize = figsize)
    sns.heatmap(pivot_table, annot = True, fmt = 'd', cmap = 'flare')
    plt.xlabel(columns, fontsize = 12)
    plt.ylabel(index, fontsize = 12)
    plt.title(title, fontsize = 16)
    plt.show()

def plot_image_with_preprocessing(metadata, method = 'grayscale'):
    """
    Displays an image with a specified preprocessing method.
    
    Parameters:
        metadata (pd.DataFrame): The metadata containing image paths.
        method (str): Preprocessing method ('grayscale', 'contrast', or 'brightness_contrast').
        
    Raises:
        ValueError: If the method is not 'grayscale', 'contrast', or 'brightness_contrast'.
        
    Returns:
        None
    """
    if method not in ['grayscale', 'contrast', 'brightness_contrast']:
        raise ValueError("Method must be 'grayscale', 'contrast', or 'brightness_contrast'.")
    
    random_index = random.choice(metadata.index)
    
    image_path = metadata.loc[random_index, 'path_to_image']
    
    original_image = cv2.imread(image_path)
    
    if method == 'grayscale':
        preprocessed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        cmap = 'gray'  
    elif method == 'contrast':
        preprocessed_image = cv2.convertScaleAbs(original_image, alpha=1.2, beta=10)
        cmap = None  
    elif method == 'brightness_contrast':
        preprocessed_image = cv2.convertScaleAbs(original_image, alpha=1.2, beta=20)
        cmap = None 
    
    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
    
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off') 
    
    if method == 'grayscale':
        axes[1].imshow(preprocessed_image, cmap=cmap)
    else:
        axes[1].imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"{method.replace('_', ' ').capitalize()} Image")
    axes[1].axis('off') 
    
    plt.tight_layout()
    plt.show()

def Training_and_Validation_Metrics_Plot(history, f1_scores):
    """
    Plots the training and validation metrics including loss and F1 score.
    
    Parameters:
        history (dict): The history object from model training.
        f1_scores (dict): Dictionary containing train and validation F1 scores.
        
    Returns:
        None
    """
    print("History keys:", history.history.keys())

    train_loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    
    train_F1 = f1_scores['train_f1_scores']
    val_F1 = f1_scores['val_f1_scores']
    
    epochs = list(range(1, len(train_loss) + 1))

    fig = make_subplots(rows = 1, cols = 2, shared_yaxes = True, column_widths = [1, 1],  subplot_titles = ["Loss", "F1 Score"], horizontal_spacing = 0.1)

    fig.add_trace(go.Scatter(x = epochs, y = train_loss, mode = 'lines', name = 'Train', line = dict(color = '#FF6692'), showlegend = True), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = epochs, y = val_loss, mode = 'lines', name = 'Validation', line = dict(color = '#B00068', dash = 'dash'), showlegend = True), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = epochs, y = train_F1, mode = 'lines', name = 'Train', line = dict(color = '#FF6692'), showlegend = False), row = 1, col = 2)
    fig.add_trace(go.Scatter(x = epochs, y = val_F1, mode = 'lines', name = 'Validation', line = dict(color = '#B00068', dash = 'dash'), showlegend = False), row = 1, col = 2)

    fig.update_layout(title = 'Training and Validation Loss & F1 Score', xaxis_title = 'Epochs', template = 'plotly_white', showlegend = True, height = 600, title_x = 0.5, plot_bgcolor = 'white', paper_bgcolor = 'white')

    fig.update_xaxes(title_text = 'Epochs', title_font = dict(color = 'black'))
    fig.update_yaxes(title_text = 'Loss', row = 1, col = 1, title_font = dict(color = 'black'))
    fig.update_yaxes(title_text = 'F1 Score', row = 1, col = 2, title_font = dict(color = 'black'))

    fig.show()

def model_evaluation_plots(labels_test, test_predictions):
    """
    Visualizes model evaluation plots including ROC, precision-recall, and calibration curves.
    
    Parameters:
        labels_test (np.array): True labels for the test set.
        test_predictions (np.array): Predictions made by the model on the test set.
        
    Returns:
        None
    """
    fpr, tpr, _ = roc_curve(labels_test, test_predictions)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(labels_test, test_predictions)

    prob_true, prob_pred = calibration_curve(labels_test, test_predictions, n_bins = 10)

    fig = make_subplots(rows = 1, cols = 3, subplot_titles = ("ROC Curve", "Precision-Recall Curve", "Calibration Curve"))

    roc_trace = go.Scatter(x = fpr, y = tpr, mode = 'lines', name = f'ROC curve (AUC = {roc_auc:.2f})', line = dict(color = '#FF6692'))
    roc_random_line = go.Scatter(x = [0, 1], y = [0, 1], mode = 'lines', name=  'Random classifier', line = dict(color = '#B00068', dash = 'dash'))
    fig.add_trace(roc_trace, row = 1, col = 1)
    fig.add_trace(roc_random_line, row = 1, col = 1)

    pr_trace = go.Scatter(x = recall, y = precision, mode = 'lines', name = 'Precision - Recall curve', line = dict(color = '#FF6692'))
    fig.add_trace(pr_trace, row = 1, col = 2)

    calibration_trace = go.Scatter(x = prob_pred, y = prob_true, mode = 'markers+lines', name = 'Calibration Curve', line = dict(color = '#FF6692'))
    calibration_random_line = go.Scatter(x = [0, 1], y = [0, 1], mode = 'lines', name = 'Perfectly Calibrated', line = dict(color = '#B00068', dash = 'dash'))
    fig.add_trace(calibration_trace, row = 1, col = 3)
    fig.add_trace(calibration_random_line, row = 1, col = 3)

    fig.update_layout(title = "Binary Classification Model Evaluation", showlegend = True, height = 500, width = 1350, plot_bgcolor = 'white')

    fig.update_xaxes(title = "False Positive Rate", showgrid = True, gridcolor = 'lightgrey', row = 1, col = 1)
    fig.update_yaxes(title = "True Positive Rate", showgrid = True, gridcolor = 'lightgrey', row = 1, col = 1)

    fig.update_xaxes(title = "Recall", showgrid = True, gridcolor = 'lightgrey', row = 1, col = 2)
    fig.update_yaxes(title = "Precision", showgrid = True, gridcolor = 'lightgrey', row = 1, col = 2)

    fig.update_xaxes(title = "Mean Predicted Probability", showgrid = True, gridcolor = 'lightgrey', row = 1, col = 3)
    fig.update_yaxes(title = "Fraction of Positives", showgrid = True, gridcolor = 'lightgrey', row = 1, col = 3)

    fig.show()

def confusion_matrix_multi_class(model, test_data, test_labels, class_names):
    """
    Generates and visualizes the confusion matrix for a multi-class classification problem.

    Parameters:
        model (keras.Model): A trained Keras model used for making predictions on the test data.
        test_data (np.array): Input data for predictions.
        test_labels (np.array): True labels for the test data in one-hot encoded format.
        class_names (list): List of class names corresponding to the label indices.

    Returns:
        None
    """
    predictions = model.predict(test_data)

    predicted_classes = np.argmax(predictions, axis = 1)
    true_classes = np.argmax(test_labels, axis = 1)

    cm = confusion_matrix(true_classes, predicted_classes)

    plt.figure(figsize = (10, 7))
    sns.heatmap(cm, annot = True, fmt = "d", cmap = "flare", xticklabels = class_names, yticklabels = class_names)
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# --------------------------------------------------
# Image Processing Functions
# --------------------------------------------------

def compute_image_hash(image_path):
    """
    Computes a perceptual hash for an image based on its grayscale pixel values.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        str: Hexadecimal representation of the computed image hash.
        None: If an error occurs during processing.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
            avg_pixel = sum(pixels) / len(pixels)
            bits = ''.join(['1' if pixel > avg_pixel else '0' for pixel in pixels])
            hex_hash = f"{int(bits, 2):016x}"
            return hex_hash
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def find_duplicate_images(metadata):
    """
    Identifies duplicate images based on perceptual hashing.

    Parameters:
        metadata (pd.DataFrame): A DataFrame containing the column 'path_to_image' with paths to image files.

    Returns:
        dict: A dictionary where keys are hash values and values are lists of image paths 
              corresponding to duplicate images.
    """
    hash_dictionary = defaultdict(list)

    for image_path in metadata['path_to_image']:
        image_hash = compute_image_hash(image_path)
        if image_hash:
            hash_dictionary[image_hash].append(image_path)

    duplicate_images = {hash_value: paths for hash_value, paths in hash_dictionary.items() if len(paths) > 1}
    
    return duplicate_images

seen_paths = set()

def keep_one_image(image_path):
    """
    Keeps the first occurrence of each image path and filters out duplicates.

    Parameters:
        image_path (str): The path of an image.

    Returns:
        str: The image path if it hasn't been seen before.
        None: If the image path is a duplicate.
    """
    if image_path not in seen_paths:
        seen_paths.add(image_path)
        return image_path
    else:
        return None  

def load_images(image_paths, image_size = (50, 50)):
    """
    Loads and preprocesses images from the specified paths.
    
    Parameters:
        image_paths (list): List of paths to the images to be loaded.
        image_size (tuple, optional): Desired size for resizing images (width, height). Defaults to (50, 50).
        
    Returns:
        np.ndarray: Array of preprocessed images.
    """
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, image_size)  
        img = img / 255.0    
        images.append(img)
    
    return np.array(images)

def apply_preprocessing(X_train, method):
    """
    Applies preprocessing techniques to the images.
    
    Parameters:
        X_train (np.ndarray): Array of images to be preprocessed.
        method (str): Preprocessing method ('grayscale', 'contrast', or 'brightness_contrast').
        
    Returns:
        tf.Tensor: Tensor of preprocessed images.
    """
    processed_images = []

    for image in X_train:
        if not isinstance(image, np.ndarray):
            print(f"Skipping non-numpy image")
            continue
        
        if len(image.shape) != 3:
            print(f"Skipping image with invalid shape: {image.shape}")
            continue

        image = image.astype(np.float32)

        if method == 'grayscale':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = np.expand_dims(gray, axis = -1)  
            processed_images.append(gray)

        elif method == 'contrast':
            contrast = cv2.convertScaleAbs(image, alpha = 1.2, beta = 10)  
            processed_images.append(contrast)

        elif method == 'brightness_contrast':
            brightness_contrast = cv2.convertScaleAbs(image, alpha = 1.2, beta = 20)
            processed_images.append(brightness_contrast)

    processed_images = np.array(processed_images)
    processed_images_tensor = tf.convert_to_tensor(processed_images, dtype = tf.float32)

    return processed_images_tensor

# --------------------------------------------------
# Modelling
# --------------------------------------------------

def model_building_function_1(hp):
    """
    Builds a Keras Sequential model with hyperparameter tuning for binary classification.

    Parameters:
        hp (keras_tuner.HyperParameters): An object to define and sample hyperparameters.

    Returns:
        keras.Model: A compiled Keras model ready for training.
    """
    model = models.Sequential()
    
    model.add(layers.Conv2D(filters = hp.Int('conv_1_filters', min_value = 32, max_value = 128, step = 32), kernel_size = hp.Choice('conv_1_kernel_size', values = [3, 5]), activation = 'relu', input_shape = (50, 50, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(filters = hp.Int('conv_2_filters', min_value = 64, max_value = 256, step = 64), kernel_size = hp.Choice('conv_2_kernel_size', values = [3, 5]), activation = 'relu'))
    
    model.add(layers.Conv2D(filters = hp.Int('conv_3_filters', min_value = 128, max_value = 512, step = 128), kernel_size = hp.Choice('conv_3_kernel_size', values = [3, 5]), activation = 'relu'))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(units = hp.Int('dense_1_units', min_value = 128, max_value = 512, step = 128), activation = 'relu'))
    model.add(layers.Dropout(rate = hp.Float('dropout_rate', min_value = 0.3, max_value = 0.7, step = 0.1)))
    
    model.add(layers.Dense(units = hp.Int('dense_2_units', min_value = 64, max_value = 256, step = 64), activation = 'relu'))
    
    model.add(layers.Dense(1, activation = 'sigmoid'))
    
    model.compile(optimizer = Adam(learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])), loss = 'binary_crossentropy', metrics = ['recall', 'precision'])
    
    return model

def transfer_learning(base_model):
    """
    Constructs a transfer learning model using a pre-trained base model.
    
    Parameters:
        base_model (tf.keras.Model): A pre-trained Keras model.
        
    Returns:
        tf.keras.Model: Transfer learning model.
    """
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    
    x = Dense(256, activation = 'relu')(x)  
    x = Dropout(0.3)(x) 
    
    x = Dense(256, activation = 'relu')(x)  
    
    x = Dense(128, activation = 'relu')(x) 
    x = Dropout(0.4)(x)  
    
    x = Dense(64, activation = 'relu')(x) 
    x = Dropout(0.4)(x)  
    
    x = Dense(32, activation = 'relu')(x)  
    
    output = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs = base_model.input, outputs = output)
    
    model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'binary_crossentropy', metrics = ['recall', 'precision'])
    
    return model

def model_building_function_2(hp):
    """
    Builds a Keras Sequential model with hyperparameter tuning for multi-class classification.

    Parameters:
        hp (keras_tuner.HyperParameters): An object to define and sample hyperparameters.

    Returns:
        keras.Model: A compiled Keras model ready for training.
    """
    model = models.Sequential()

    model.add(layers.Conv2D(filters = hp.Int('conv_1_filters', min_value = 32, max_value = 128, step = 16), kernel_size = hp.Choice('conv_1_kernel_size', values = [3, 5]), activation = 'relu', input_shape = (50, 50, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(filters = hp.Int('conv_2_filters', min_value = 32, max_value = 128, step = 16), kernel_size = hp.Choice('conv_2_kernel_size', values = [3, 5]), activation  = 'relu'))

    model.add(layers.Conv2D(filters = hp.Int('conv_3_filters', min_value = 64, max_value = 256, step = 32), kernel_size = hp.Choice('conv_3_kernel_size', values = [3, 5]), activation = 'relu'))

    model.add(layers.Flatten())

    model.add(layers.Dense(units = hp.Int('dense_1_units', min_value = 64, max_value = 256, step = 32), activation = 'relu'))
    model.add(layers.Dropout(rate = hp.Float('dropout_rate', min_value = 0.2, max_value = 0.6, step = 0.1)))

    model.add(layers.Dense(units = hp.Int('dense_2_units', min_value = 32, max_value = 128, step = 16), activation = 'relu'))

    model.add(layers.Dense(8, activation = 'softmax'))

    model.compile(optimizer = Adam(learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])), loss = 'categorical_crossentropy', metrics = ['recall', 'precision'])

    return model

def transfer_learning_multi_class(base_model):
    """
    Builds a transfer learning model for multi-class classification using a pre-trained base model.

    Parameters:
        base_model (keras.Model): A pre-trained Keras model to be used as the base. 
                                  Its layers are frozen during training.

    Returns:
        keras.Model: A compiled Keras model with additional dense layers for classification.
    """
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    
    x = Dense(256, activation = 'relu')(x)  
    x = Dropout(0.3)(x) 
    
    x = Dense(256, activation = 'relu')(x)  
    
    x = Dense(128, activation = 'relu')(x) 
    x = Dropout(0.4)(x)  
    
    x = Dense(64, activation = 'relu')(x) 
    x = Dropout(0.4)(x)  
    
    x = Dense(32, activation = 'relu')(x)  
    
    output = Dense(8, activation = 'softmax')(x)

    model = Model(inputs = base_model.input, outputs = output)
    
    model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['recall', 'precision'])
    
    return model

def model_building_function_3(hp):
    """
    Builds a Keras Functional API model with hyperparameter tuning.

    The model takes two inputs: image data and binary labels, processes them separately, 
    and combines their outputs for multi-class classification.

    Parameters:
        hp (keras_tuner.HyperParameters): An object to define and sample hyperparameters.

    Returns:
        keras.Model: A compiled Keras Functional API model ready for training.
    """
    image_input = Input(shape = (50, 50, 3), name = 'image_input')

    x = Conv2D(filters = hp.Int('conv_1_filters', min_value = 32, max_value = 128, step = 32), kernel_size = hp.Choice('conv_1_kernel_size', values = [3, 5]), kernel_regularizer = l2(hp.Float('conv_1_l2', min_value = 0.01, max_value = 0.05, step = 0.01)))(image_input)
    x = LeakyReLU(alpha = 0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)

    x = Conv2D(filters = hp.Int('conv_2_filters', min_value = 64, max_value = 256, step = 64), kernel_size = hp.Choice('conv_2_kernel_size', values = [3, 5]), kernel_regularizer = l2(hp.Float('conv_2_l2', min_value = 0.01, max_value = 0.05, step = 0.01)))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)

    x = Conv2D(filters = hp.Int('conv_3_filters', min_value = 128, max_value = 512, step = 128), kernel_size = hp.Choice('conv_3_kernel_size', values = [3, 5]), kernel_regularizer = l2(hp.Float('conv_3_l2', min_value = 0.01, max_value = 0.05, step = 0.01)))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    x = Dense(units = hp.Int('dense_1_units', min_value = 128, max_value = 512, step = 128), activation = 'relu', kernel_regularizer = l2(hp.Float('dense_1_l2', min_value = 0.01, max_value = 0.05, step = 0.01)))(x)
    x = Dropout(rate = hp.Float('dropout_1_rate', min_value = 0.3, max_value = 0.7, step = 0.1))(x)

    x = Dense(units = hp.Int('dense_2_units', min_value = 64, max_value = 256, step = 64), activation = 'relu', kernel_regularizer = l2(hp.Float('dense_2_l2', min_value = 0.01, max_value = 0.05, step = 0.01)))(x)

    binary_input = Input(shape = (1,), name = 'binary_input')

    binary_path = Dense(units = hp.Int('binary_dense_1_units', min_value = 16, max_value = 64, step = 16), activation = 'relu', kernel_regularizer = l2(hp.Float('binary_dense_1_l2', min_value = 0.01, max_value = 0.05, step = 0.01)))(binary_input)
    binary_path = BatchNormalization()(binary_path)
    binary_path = Dropout(rate = hp.Float('binary_dropout_rate', min_value = 0.1, max_value = 0.4, step = 0.1))(binary_path)
    binary_path = Dense(units = hp.Int('binary_dense_2_units', min_value = 8, max_value = 32, step = 8), activation = 'relu', kernel_regularizer = l2(hp.Float('binary_dense_2_l2', min_value = 0.01, max_value = 0.05, step = 0.01)))(binary_path)

    combined = Concatenate()([x, binary_path])

    final_output = Dense(units = 8, activation = 'softmax', name = 'output')(combined)

    functional_API = Model(inputs = [image_input, binary_input], outputs = final_output)

    functional_API.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]), decay = 1e-6), loss = 'categorical_crossentropy', metrics = ['recall', 'precision'])

    return functional_API

# --------------------------------------------------
# Evaluation Functions
# --------------------------------------------------

def obtain_f1_score(history, i):
    """
    Computes the F1-score for the model's training and validation phases.
    
    Parameters:
        history (dict): The history object from model training.
        i (int): Index to access specific metrics in the history (0 for default).
        
    Returns:
        dict: Dictionary containing train and validation F1 scores.
    """
    if i == 0:
        train_precision = history['precision']
        train_recall = history['recall']
        val_precision = history['val_precision']
        val_recall = history['val_recall']
    else:
        train_precision = history[f'precision_{i}']
        train_recall = history[f'recall_{i}']
        val_precision = history[f'val_precision_{i}']
        val_recall = history[f'val_recall_{i}']

    train_f1_scores = [
        2 * (p * r) / (p + r) if (p + r) > 0 else 0 
        for p, r in zip(train_precision, train_recall)]
    val_f1_scores = [
        2 * (p * r) / (p + r) if (p + r) > 0 else 0 
        for p, r in zip(val_precision, val_recall)]

    return {'train_f1_scores': train_f1_scores, 'val_f1_scores': val_f1_scores}

def evaluate_model(true_labels, predictions):
    """
    Evaluates the model's performance using a confusion matrix and classification report.
    
    Parameters:
        true_labels (np.array): True labels for the test set.
        predictions (np.array): Predictions made by the model.
        
    Returns:
        None
    """
    Classification_Report = classification_report(true_labels, predictions)
    
    Confusion_Matrix = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize = (10, 7))
    sns.heatmap(Confusion_Matrix, annot = True, fmt = "d", cmap = "flare")
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
    
    print(Classification_Report)

def evaluate_each_class(model, test_data, test_labels, class_names):
    """
    Evaluates model performance for each class and prints a classification report.

    Parameters:
        model (keras.Model): A trained Keras model used for making predictions.
        test_data (np.array): Input data for predictions.
        test_labels (np.array): True labels for the test data in one-hot encoded format.
        class_names (list): List of class names corresponding to the label indices.

    Returns:
        None
    """
    predictions = model.predict(test_data)

    predicted_classes = np.argmax(predictions, axis = 1)
    true_classes = np.argmax(test_labels, axis = 1)

    report = classification_report(true_classes, predicted_classes, target_names = class_names)
    print(report)

