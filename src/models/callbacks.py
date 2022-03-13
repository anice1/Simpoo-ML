import datetime
import tensorflow as tf
import os

time_stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')

def learning_rate_scheduler(rate=3e-1*10):
    """
    A learning rate callback
    """
    
    lr = tf.keras.callbacks.LearningRateScheduler(lambda epochs: rate ** (epochs/20))
    return lr
  
def model_checkpoint_callback(experiment_name:str, path='models/callbacks'):
    """Creates a ModelCheckpoint callback 

    Args:
        experiment_name (str): Name of current experiment (e.g 'base_model')
        path: file path where checkpoint is saved
    Returns:
        _type_: _description_
    """
    checkpoint_path = os.path.join(path,time_stamp+'_'+experiment_name.replace(' ','_').lower())
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                           verbose=1,
                                                           save_best_weights_only=True)
    print('ModelCheckpoint callback saved to: ', checkpoint_path)
    return checkpoint_callback

def tensorboard_callback(experiment_name:str, path='models/callbacks'):
    """Creates a TensorBoard callback 

    Args:
        log_dir (str): Name of directory to store the callback object
        experiment_name (str): Name of current experiment (e.g 'base_model')

    Returns:
        dir: returns a log directory containing the tensorboard callbacks validation and training logs
    """
    directory = os.path.join(path, time_stamp+'_'+experiment_name.replace(' ','_').lower())

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=directory)
    print('Tensorboard callback saved to:', 'Callbacks/'+directory)
    return tensorboard_callback