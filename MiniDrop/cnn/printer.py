import tensorflow as tf

def print_model_weights(model):
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            print(f"Weights of Layer '{layer.name}':")
            for i, weight in enumerate(weights):
                print(f"Weight {i+1}:")
                print(weight)
        else:
            print(f"No weights found for Layer '{layer.name}'")

# Load the pre-trained model from an H5 file
model_path = '/home/mabon/temp333/model/best_model.h5'
model = tf.keras.models.load_model(model_path)

# Print the weights of all layers in the model
print_model_weights(model)
