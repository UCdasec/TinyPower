import tensorflow as tf

def count_parameters(model):
    return sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

# Load the first model
model1 = tf.keras.models.load_model("model1.h5")

# Load the second model
model2 = tf.keras.models.load_model("model2.h5")

# Calculate the total number of parameters for each model
params_model1 = count_parameters(model1)
params_model2 = count_parameters(model2)

# Calculate the percentage change in the total number of parameters
percentage_change = ((params_model2 - params_model1) / params_model1) * 100

print(f"Total parameters in Model 1: {params_model1}")
print(f"Total parameters in Model 2: {params_model2}")
print(f"Percentage change in parameters: {percentage_change:.2f}%")
