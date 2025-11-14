"""
Script to benchmark PointNet implementation.

Author: pranjal
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pointnet import create_pointnet_model
from plotting import plot_learning_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import itertools
import tqdm
import os


def hyperparameter_tuning(batch_size_num: int,learning_rate_num: int):
    """Training with a specific batch_size and learning_rate combination

    Args:
        batch_size_num (int): batch size 
        learning_rate_num (int): learning rate
    Return:
        (float): best validation accuracy for the specific combination
    """
    print(f"\n>>> Training with batch size = {batch_size_num} and learning rate = {learning_rate_num}")

    train_features = np.load('/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data/16O_size800_train_features.npy')
    train_labels = np.load('/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data/16O_size800_train_labels.npy')
    print("Training data shape:", train_features.shape)
    train_ds = tf.data.Dataset.from_tensor_slices((train_features[:,:,:3], train_labels))
    train_ds = train_ds.batch(batch_size=batch_size_num, drop_remainder=False)

    val_features = np.load('/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data/16O_size800_val_features.npy')
    val_labels = np.load('/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data/16O_size800_val_labels.npy')
    val_ds = tf.data.Dataset.from_tensor_slices((val_features[:,:,:3], val_labels))
    val_ds = val_ds.batch(batch_size=batch_size_num, drop_remainder=True)

    
    #early stopping
    early_stopping = EarlyStopping(
        monitor = "val_sparse_categorical_accuracy",
        mode = "max",
        patience = 10,
        restore_best_weights = True,
    )

    # build and train event-wise classification model and plot learning curve
    model = create_pointnet_model(num_points=800, 
                          num_classes=5, 
                          num_dimensions=3, #for changing number of features
                          is_regression=False,
                          is_pointwise_prediction=False)
    #model.summary()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate_num),
                  metrics=["sparse_categorical_accuracy"])
    

    #checkpointing
    history = model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=[early_stopping],verbose=2) #early stopping

    best_val_acc = np.max(history.history["val_sparse_categorical_accuracy"])
    print(f"Best val_accuracy={best_val_acc:.4f}")   
    return best_val_acc


def run_experiment():
    """Finding the best batch size and learning rate combination 

    Returns:
        (tuple): best parameters (batch_size, learning_rate)
    """
    batch_options = [128,256]
    lr_options = [3e-6,5e-6,6e-6,7.5e-6]
    results = {}

    for bs, lr in itertools.product(batch_options,lr_options):
        val_acc = hyperparameter_tuning(bs,lr)
        results[(bs,lr)] = val_acc

    opt_params = max(results, key=results.get)
    print(f"\n Best parameter pair is {opt_params} with the accuracy of {results[opt_params]}")

    return opt_params


def train_best_model():
    """Training the model with the best parameters 
    
        Returns:
        None
    """
    batch_size_num, learning_rate_num = run_experiment()
    print(f"\n>>> Training with optimised batch size = {batch_size_num} and learning rate = {learning_rate_num}")

    train_features = np.load('/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data/16O_size800_train_features.npy')
    train_labels = np.load('/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data/16O_size800_train_labels.npy')
    print("Training data shape:", train_features.shape)
    train_ds = tf.data.Dataset.from_tensor_slices((train_features[:,:,:3], train_labels))
    train_ds = train_ds.batch(batch_size=batch_size_num, drop_remainder=False)

    val_features = np.load('/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data/16O_size800_val_features.npy')
    val_labels = np.load('/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data/16O_size800_val_labels.npy')
    val_ds = tf.data.Dataset.from_tensor_slices((val_features[:,:,:3], val_labels))
    val_ds = val_ds.batch(batch_size=batch_size_num, drop_remainder=True)
    
    #early stopping
    early_stopping = EarlyStopping(
        monitor = "val_sparse_categorical_accuracy",
        mode = "max",
        patience = 10,
        restore_best_weights = True,
    )

    best_model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="/mnt/home/singhp19/alpha/PointNet_ATTPC/training/best_model.keras",
    monitor="val_sparse_categorical_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

    # build and train event-wise classification model and plot learning curve
    model = create_pointnet_model(num_points=800, 
                          num_classes=5, 
                          num_dimensions=3, #for changing number of features
                          is_regression=False,
                          is_pointwise_prediction=False)

    model.summary()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate_num),
                  metrics=["sparse_categorical_accuracy"])
    

    #checkpointing
    history = model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=[early_stopping,best_model_callback],verbose=2) #early stopping
    plot_learning_curve(history, '/mnt/home/singhp19/alpha/PointNet_ATTPC/learning-curve.png',batch_size_num,learning_rate_num)

    best_epoch = np.argmax(history.history["val_sparse_categorical_accuracy"]) + 1
    best_val_acc = np.max(history.history["val_sparse_categorical_accuracy"])
    print(f"Best model was at epoch {best_epoch} with val_accuracy={best_val_acc:.4f}") 


def load_classfication_model():
    """Evaluating the performance on the model with best hyperparameters 
       
        Returns:
        None
    """
    tf.keras.backend.clear_session() 
    test_features = np.load("/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data/16O_size800_test_features.npy")
    test_features = test_features[:,:,:3]
    test_labels = np.load("/mnt/research/attpc/e20020/Pointet_MLclassification/engine_training_data/16O_size800_test_labels.npy")

    print("Shape of test_features:", test_features.shape)

    model = create_pointnet_model(num_points=800, 
                          num_classes=5, 
                          num_dimensions=3, #for changing number of features
                          is_regression=False,
                          is_pointwise_prediction=False)
    
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=5e-06),
                  metrics=["sparse_categorical_accuracy"])
    
    model.summary()
    loss, accuracy_d = model.evaluate(test_features,test_labels,verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * accuracy_d))
    print("Unique label values:", np.unique(test_labels))

    model = tf.keras.models.load_model("/mnt/home/singhp19/alpha/PointNet_ATTPC/training/engine_w0_noise/best_model.keras")
    loss, accuracy_d = model.evaluate(test_features,test_labels,verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * accuracy_d))
    
    y_pred = model.predict(test_features)
    predicted_classes = np.argmax(y_pred, axis=1)

    f1 = f1_score(test_labels, predicted_classes, average='weighted')
    f1_cl2 = f1_score(test_labels, predicted_classes, labels=[2], average='weighted')
    cf_matrix = confusion_matrix(test_labels, predicted_classes)
    
    print(f"F1 Score: {f1}")
    print(f"F1 Score Class 2: {f1_cl2}")
    print("Confusion Matrix:")
    #print(cm)

    group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(5,5)
    class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]  # Change to your labels

    report = classification_report(test_labels, predicted_classes, target_names=class_names)
    print(report)

    svm = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Reds')
    svm.set_xlabel("Predicted Labels")
    svm.set_ylabel("True Labels")
    svm.set_xticklabels(class_names)
    svm.set_yticklabels(class_names, rotation=0) 
    plt.savefig("/mnt/home/singhp19/alpha/PointNet_ATTPC/confusion_matrix.png")


if __name__ == '__main__':
    tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    train_best_model()
    load_classfication_model()