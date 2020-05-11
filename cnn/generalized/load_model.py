# from cnn.nn_architecture import keras_model
from keras.engine.saving import load_model as lm
from cnn.nn_architecture.custom_loss import keras_loss_v3_nor
from cnn.nn_architecture.custom_performance_metrics import keras_accuracy, \
    accuracy_asloss, accuracy_asproduction, keras_binary_accuracy


def load_model(config):

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



# predict_patch_and_save_results(
#         model, 'val_set', df_val, True,
#         BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, path_results,
#         True)

