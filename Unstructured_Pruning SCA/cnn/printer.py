# from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from one_cycle_lr import OneCycleLR
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.sparsity.keras import UpdatePruningStep
from tensorflow.keras import backend as K
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
#model.summary()
# Print the weights of all layers in the model
#print_model_weights(model)
