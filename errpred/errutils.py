from functools import partial
import numpy as np 


g_name_counter = 1

def copy_model(model):
    global g_name_counter
    """ Return a copy of the keras model """
    from keras.models import clone_model
    model_copy = clone_model(model)
    model_copy.set_weights(model.get_weights())
    model_copy.name = "{}_{}".format(model_copy.name, g_name_counter)
    g_name_counter += 1

    model_copy.compile("adam", loss=model.loss)
    return model_copy

def to_shape(a):
    if isinstance(a, list): return [k.shape for k in a]
    else: return a.shape

def normalize_data(*discrete_series):
    def transform_array(a):
        if len(a.shape) == 1: return a.reshape(-1,1)
        else: return a

    def transform_serie(serie):
        serie = serie if isinstance(serie, list) else [serie]
        serie = list(map(transform_array, serie))
        return serie

    return list(map(transform_serie, discrete_series))

def to_tensor_shape(shape):
    if len(shape) == 1: return (1,)
    else: return shape[1:]

def split_train_test(data, validation_fraction):
    split_index = int((1.0-validation_fraction) * len(data))    
    train, test = data[:split_index], data[split_index:]
    return train, test

def split_train_validation(x_train : [], y_train : [], validation_fraction):
    split_index = int((1.0-validation_fraction) * len(x_train[0]))    

    x_train, x_validation = [tx[:split_index] for tx in x_train], [tx[split_index:] for tx in x_train]
    y_train, y_validation = [ty[:split_index] for ty in y_train], [ty[split_index:] for ty in y_train]
    return x_train, y_train, x_validation, y_validation


def random_select_samples(x, y, dropout_factor=0.3):
    # Get the sample indices to include in training
    NUM_SAMPLES = int(len(x[0]) * (1-dropout_factor))
    sample_indices = np.arange(NUM_SAMPLES)
    np.random.shuffle(sample_indices)
    sample_indices = sample_indices[:NUM_SAMPLES]

    # Remap x and y to only include the selected samples
    x = [tx[sample_indices] for tx in x]
    y = [ty[sample_indices] for ty in y]

    return x,y


def fit_model(model, x, y, max_epochs=500, patience=25, batch_size=256, shuffle=False, validation_xy=None, validation_split=None, best_model_opt=False):
    import keras, os
    from keras.models import load_model
    from tempfile import NamedTemporaryFile
    from keras_tqdm import TQDMCallback
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping
    import io
    
    # Configure callbacks
    keras_callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, verbose=0, mode='auto'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10)]
    if best_model_opt: keras_callbacks.append(keras.callbacks.ModelCheckpoint("temp.model", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto'))

    try:
        # Fit the model
        if validation_xy is None:
            model.fit(x, y, epochs=max_epochs, batch_size=batch_size, callbacks=keras_callbacks, shuffle=shuffle, verbose=1, validation_split=validation_split)
        else:
            model.fit(x, y, epochs=max_epochs, batch_size=batch_size, callbacks=keras_callbacks, shuffle=shuffle, verbose=1, validation_data=validation_xy)
        if best_model_opt: model = model.load_weights("temp.model")
        return model
    finally:
        if os.path.isfile("temp.model"): os.unlink("temp.model")

def print_summary(y_test, y_pred):        
    # Calculate model statistics
    mae = np.mean(np.abs(y_pred - y_test))
    mse = np.mean(np.square(y_test - y_pred))
    mape = np.mean(np.abs((y_test - y_pred)) / y_test) * 100
    rmse = np.sqrt(np.mean(np.square(y_pred - y_test)))

    print("\nModel summary:")
    print("------------------------------------------------")
    print("MAE={:.2f}, MSE={:.2f}, MAPE={:.2f}, RMSE={:.2f}".format(mae, mse, mape, rmse))