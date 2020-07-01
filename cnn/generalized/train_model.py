from cnn import keras_utils
from cnn.nn_architecture import keras_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import numpy as np
import cnn.nn_architecture.keras_generators as gen

# extra imports for choosing GPU
import os
import tensorflow as tf

IMAGE_SIZE = 512
BATCH_SIZE = 5
BOX_SIZE = 16


def train_model(config, df_train, df_val, df_test):

    ## Import parameters
    resized_images     = config['resized_images_before_training']
    dataset            = config['dataset']
    skip_processing    = config['skip_processing_labels']
    mura_interpolation = config['mura_interpolation']
    reg_weight         = config['reg_weight']
    path_trained       = config['trained_models_path']
    path_results       = config['results_path']
    nr_epochs          = config['nr_epochs']
    pooling_operator   = config['pooling_operator']
    lr                 = config['lr']


    ## Select a custom GPU (Tensorflow v1) ====================================

    if config['gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu'][0]
        tfconfig = tf.compat.v1.ConfigProto()
        tfconfig.gpu_options.per_process_gpu_memory_fraction = config['gpu'][1]
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tfconfig))

    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    os.environ['TF_DETERMINISTIC_OPS'] = 'true'
    tf.random.set_seed(2)
    tf.keras.backend.clear_session()


    ## 1. Generate training and validation batches ============================

    # Generate training batches
    train_generator = gen.BatchGenerator(
        instances     = df_train.values,
        resized_image = resized_images,
        batch_size    = BATCH_SIZE,
        net_h         = IMAGE_SIZE,
        net_w         = IMAGE_SIZE,
        shuffle       = True,
        norm          = keras_utils.normalize,
        box_size      = BOX_SIZE,
        processed_y   = skip_processing,
        interpolation = mura_interpolation)

    # Generate validation batches
    valid_generator = gen.BatchGenerator(
        instances     = df_val.values,
        resized_image = resized_images,
        batch_size    = BATCH_SIZE,
        shuffle       = True,
        net_h         = IMAGE_SIZE,
        net_w         = IMAGE_SIZE,
        box_size      = BOX_SIZE,
        norm          = keras_utils.normalize,
        processed_y   = skip_processing,
        interpolation = mura_interpolation)


    ## 2. Build the model =====================================================

    # Configure the network architecture
    model = keras_model.build_model(reg_weight)
    model.summary()

    # Add ADAM optimizer and accuracy metrics
    model = keras_model.compile_model_accuracy(model, lr, pooling_operator)

    # Create early stopping criteria
    early_stop = EarlyStopping(
        monitor   = 'val_loss',
        min_delta = 0.001,
        patience  = 60,
        mode      = 'min',
        verbose   = 1)

    # Model name extension
    filepath = f'{path_trained}_best_model_{dataset}-'+'{epoch:02d}-{val_loss:.2f}.hdf5'

    # Save the best model epoch
    checkpoint = ModelCheckpoint(
        filepath        = filepath,
        monitor         = 'val_loss',
        verbose         = 2,
        save_best_only  = True,
        mode            = 'min'
    )

    # Configure the dynamic learning rate
    lrate = LearningRateScheduler(keras_model.step_decay, verbose=1)

    print("Training -- STEPS:")
    print(len(df_train) // BATCH_SIZE)
    print(train_generator.__len__())
    print()


    ## 3. Training the network ================================================

    history = model.fit_generator(
        generator        = train_generator,
        steps_per_epoch  = train_generator.__len__(),
        epochs           = nr_epochs,
        validation_data  = valid_generator,
        validation_steps = valid_generator.__len__(),
        verbose          = 1,
        callbacks        = [checkpoint, lrate]
    )


    ## 4. Return results ======================================================

    filepath = f'{path_trained}model_{dataset}-'+'{val_loss:.2f}.hdf5'
    model.save(filepath)

    print("history")
    print(history.history)
    print(history.history['keras_accuracy'])

    # Save training history to a file
    np.save(f'{path_results}train_info_{dataset}.npy', history.history)

    # Save training settings to a file
    settings = np.array({
    'lr: ':                lr,
    'reg_weight: ':        reg_weight,
    'pooling_operator: ':  pooling_operator})
    np.save(path_results + 'train_settings.npy', settings)

    # Plot training and validation curves and save to a file
    keras_utils.plot_train_validation(
        history.history['loss'],
        history.history['val_loss'],
        'train loss', 'validation loss', 'loss_' + dataset,
        'loss_' + dataset, path_results)
