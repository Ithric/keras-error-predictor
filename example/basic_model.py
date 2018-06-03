from functools import partial

def __to_tensor_shape(shape):
    if len(shape) == 1: return (1,)
    else: return shape[1:]

def build_model(input_shapes : [], output_shapes : []):    
    """ Build a generic model for any input/output combination (more or  less..) """
    from keras import Model
    from keras.layers import Dense, LSTM, Input, Flatten, concatenate, Reshape
    from keras.optimizers import Adam
    print("Building model for input={}, output={}".format(input_shapes, output_shapes))

    # Remove the sample dimension
    input_shapes = list(map(__to_tensor_shape, input_shapes))
    output_shapes = list(map(__to_tensor_shape, output_shapes))

    def input_leg(input_tensor):
        leg = Dense(64)(input_tensor)
        leg = Dense(64, activation="relu")(leg)
        leg = Dense(256, activation="elu")(leg)
        leg = Dense(128, activation="elu")(leg)
        if(len(input_tensor.shape)) > 2:
            leg = Flatten()(leg)
        return leg

    all_inputs = list(map(lambda s: Input(shape=s), input_shapes))
    all_legs = list(map(input_leg, all_inputs))
    y = concatenate(all_legs) if len(all_legs)> 1 else all_legs[0]

    def output_leg(leg_y, output_shape):
        # Shape into target output shape and output the result
        open_output_shape = (output_shape[:-1]) + (-1,)
        leg_y = Reshape(open_output_shape)(leg_y)
        leg_y = Dense(output_shape[-1])(leg_y)
        return leg_y

    y = list(map(partial(output_leg, y), output_shapes))
    model = Model(inputs=all_inputs, outputs=y)        
    model.compile(optimizer=Adam(lr=0.001), loss="mae")
    return model
