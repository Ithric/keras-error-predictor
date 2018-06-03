from errpred.errutils import *
from errpred.aleatoric_error import train_error_model, create_error_model, predict_error
import numpy as np

def train_epistemic_error_model(errmodel, x, y):
    """ Train the epistemic error model
    Arguments:
        - errmodel: The epistemic error model
        - x/y: Training data. X is same as target model input, and y is the epistemic error estimate
    """
    errmodel.fit(x,y, epochs=10, batch_size=256, validation_split=0.1)

def __value_model_simulation(model, x, y, validation_xy=None):
    fit_model(model, x, y, validation_xy=validation_xy, best_model_opt=True)
    return model.predict(x)

def __error_model_simulation(model, x, y, pred_y):
    train_error_model(model, x, y, pred_y)
    return predict_error(model, x)

def create_epistemic_error_model(target_model, x, y, validation_xy=None, num_simulations=2, sample_dropout_factor=0.3):  
    assert num_simulations >= 2, "Number of simulations must be greater than or equal to 2"
    def run_simulation(simid):
        value_model, error_model = copy_model(target_model), create_error_model(target_model)
        tx, ty = random_select_samples(x, y, dropout_factor=sample_dropout_factor)
        v = __value_model_simulation(value_model, tx, ty, validation_xy=validation_xy)
        e = __error_model_simulation(error_model, tx, ty, v)
        return value_model, error_model
        
    return [run_simulation(i) for i in range(num_simulations)]

def estimate_epistemic_error(submodels, x):
    def run_predict_simulation(model_tuple):
        value_model, error_model = model_tuple
        pred_value = value_model.predict(x)
        pred_error = predict_error(error_model, x)
        return pred_value, pred_error

    # TODO: rewrite to use KL divergence ?
    val, err = zip(*map(run_predict_simulation, submodels))
    val, err = np.var(val, axis=0), np.var(err, axis=0)

    final_result = val + err**2
    return final_result