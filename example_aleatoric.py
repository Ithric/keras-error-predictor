from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import errpred as erp
from data import load_stock_timeseries_dataset
from example import basic_model as ex

VALIDATION_FRACTION = 0.15

# Load the data
x_train, y_train, x_train_validation, y_train_validation, x_test, y_test, z_test, unscaler = load_stock_timeseries_dataset(validation_fraction=VALIDATION_FRACTION, train_validation_fraction=0.25)

# Build the model, and create a copy for the error model
model = ex.build_model(erp.to_shape(x_train), erp.to_shape(y_train))
errmodel = erp.create_error_model(model)

# Train prediction model
print("Training value model")
erp.fit_model(model, x_train, y_train, validation_xy=(x_train_validation,y_train_validation), patience=30, best_model_opt=True)
x_train_pred = model.predict(x_train)
x_train_validation_pred = model.predict(x_train_validation)

# Train the error model in two stages: 1) pre-train on entire training set 2) post-fit on the validation set
# to more accurately estimate error on unseen data
print("Training error model")
erp.train_error_model(errmodel, x_train, y_train, x_train_pred)
erp.train_error_model(errmodel, x_train_validation, y_train_validation, x_train_validation_pred)
#erp.train_error_model(errmodel, x_test, y_test, model.predict(x_test), patience=75) #-- this should achieve about 68%

# Generate predictions
y_pred = model.predict(x_test)
y_pred_err = erp.predict_error(errmodel, x_test)

# Unscale everything
test_unscaler = partial(unscaler, z_test[0])
y_pred_err = test_unscaler(y_pred + y_pred_err) - test_unscaler(y_pred)
y_test = y_test[0].reshape(-1,1)
y_pred = test_unscaler(y_pred)
y_test = test_unscaler(y_test)

# mean absolute error -> standard deviation
y_pred_err = y_pred_err / np.sqrt(2.0 / np.pi)                         
y_min, y_max = y_pred - y_pred_err, y_pred + y_pred_err

# Print model statistics and plot the prediction with the error
erp.print_summary(y_test, y_pred)
k = y_test[(y_test < y_max) & (y_test > y_min)] # ] y_pred[(y_pred < y_max) & (y_pred > y_min)]
print("Points within estimated 1std err: {:.2f}%".format(100 * len(k) / len(y_test)))

plt.plot(y_pred, color="red", label="Pred")
plt.plot(y_test, color="gray", label="True")
plt.fill_between(np.arange(len(y_min)), y_min.flatten(), y_max.flatten(), color="lightskyblue")
plt.legend()
plt.show()
