import matplotlib.pyplot as plt
from example import basic_model as ex
import numpy as np
import errpred as erp
from data import load_stock_timeseries_dataset
from functools import partial

VALIDATION_FRACTION = 0.15

# Load the data
x_train, y_train, x_train_validation, y_train_validation, x_test, y_test, z_test, unscaler = load_stock_timeseries_dataset(validation_fraction=VALIDATION_FRACTION)

# Build the target (value) model
model = ex.build_model(erp.to_shape(x_train), erp.to_shape(y_train))

# Create and train the epistemic error model
errmodel = erp.create_epistemic_error_model(model, x_train, y_train, validation_xy=(x_train_validation,y_train_validation), num_simulations=9)
pred_y_epierr = erp.estimate_epistemic_error(errmodel, x_test)

# Fit the value model
erp.fit_model(model, x_train, y_train, validation_xy=(x_train_validation,y_train_validation), best_model_opt=True)
y_pred = model.predict(x_test)

# Unscale everything
y_test = y_test[0].reshape(-1,1)
test_unscaler = partial(unscaler, z_test[0])
pred_y_epierr = test_unscaler(pred_y_epierr + y_pred) - test_unscaler(y_pred)
y_pred = test_unscaler(y_pred)
y_test = test_unscaler(y_test)

# Print a model summary
erp.print_summary(y_test, y_pred)

# Plot the prediction/true values along with the predicted epistemic error
plt.plot(y_pred, color="red", label="Pred")
plt.plot(y_test, color="gray", label="True")
plt.plot(0.25*pred_y_epierr*(np.max(y_pred) / np.max(pred_y_epierr)), color="green", label="epistemic uncertainty")
plt.legend()
plt.show()
