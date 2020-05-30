# from cnn.nn_architecture import keras_model
from keras.engine.saving import load_model as lm
from cnn.nn_architecture.custom_loss import keras_loss_v3_nor
from cnn.nn_architecture.custom_performance_metrics import keras_accuracy, \
    accuracy_asloss, accuracy_asproduction, keras_binary_accuracy

# extra imports for choosing GPU
import os
import tensorflow as tf

def load_model(config):

    ## Select a custom GPU (Tensorflow v1) ====================================

    if config['gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu'][0]
        tfconfig = tf.compat.v1.ConfigProto()
        tfconfig.gpu_options.per_process_gpu_memory_fraction = config['gpu'][1]
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tfconfig))

    ## Import parameters
    # reg_weight         = config['reg_weight']
    path_trained       = config['trained_models_path']
    model_name         = config['model_name']

    # # Configure the network architecture
    # model = keras_model.build_model(reg_weight = reg_weight)

    # # Load the weights
    # model.load_weights(path_trained + model_name)

    model = lm(path_trained + model_name, custom_objects = {
                'keras_loss_v3_nor':      keras_loss_v3_nor,
                'keras_accuracy':         keras_accuracy,
                'keras_binary_accuracy':  keras_binary_accuracy,
                'accuracy_asloss':        accuracy_asloss,
                'accuracy_asproduction':  accuracy_asproduction})

    return model
