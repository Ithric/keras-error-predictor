from errpred.errutils import *

NUM_SAMPLES = 32
PVAL = 1/NUM_SAMPLES    

def standard_error_loss(pred_mean, y_true, y_pred):
    import keras.backend as K
    delta = y_true - pred_mean
    new_pred_abs_err = y_pred*(1 - PVAL) + K.abs(delta)*PVAL

    return K.abs(y_pred - new_pred_abs_err)

def create_error_model(original_model):
    from keras.layers import Input
    from keras.models import Model
    errmodel = copy_model(original_model)

    # Input for predicted input to be used by the loss function   
    pred_mean = Input(shape=to_tensor_shape(original_model.output_shape))

    # stuff
    err_y = errmodel(errmodel.inputs)
    outer_model = Model(inputs=errmodel.inputs+[pred_mean], outputs=err_y)

    outer_model.compile(optimizer="adam", loss=partial(standard_error_loss, pred_mean))
    return outer_model

def train_error_model(errmodel, x, y, pred_y, patience=100):
    # Fit the model
    fit_model(errmodel, x+[pred_y], y, validation_split=0.10, patience=patience, best_model_opt=True)

def predict_error(errmodel, x):    
    y_pred_shape = (len(x[0]),1)
    return np.abs(errmodel.predict(x+[np.zeros(shape=y_pred_shape)]))
