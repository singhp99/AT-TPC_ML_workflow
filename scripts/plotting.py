import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_learning_curve(history, filename: str,batch_size: int,learning_rate: int):
    """Plotting the learning curve

    Args:
        history (keras.callbacks.History): History object from model.fit
        filename (str): path for saving the learning curve
        batch_size (int): batch size
        learning_rate (int): learning rate
        
    Returns:
        None
    """
    plt.figure(figsize=(11, 6), dpi=100)
    plt.plot(np.array(history.history['loss'])/batch_size, 'o-', label='Training Loss')
    plt.plot(np.array(history.history['val_loss'])/batch_size, 'o:', color='r', label='Validation Loss')
    plt.legend(loc='best')
    plt.title('Learning Curve')
    plt.suptitle(f"Batch size: {batch_size}, Learning rate: {learning_rate}", y=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(history.history['loss']), 10), range(1, len(history.history['loss']) + 1, 10))
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(filename) 
    
    

