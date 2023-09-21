import tensorflow as tf
import argparse
import os
import pathlib
import sys

# Load the model from an h5 file
def load_model_from_h5(file_path):
    model = tf.keras.models.load_model(file_path)
    return model
# Calculate FLOPs for the given CNN model
def calculate_flops(model):
    total_flops = 0
    
    for layer in model.layers:
        layer_type = type(layer).__name__

        if layer_type == 'Conv2D':
            kernel_flops = 2 * layer.filters * layer.kernel_size[0] * layer.kernel_size[1]
            output_shape = layer.output_shape[1:]
            flops = kernel_flops * output_shape[0] * output_shape[1] * output_shape[2]
            total_flops += flops

        elif layer_type == 'Dense':
            flops = 2 * layer.input_shape[-1] * layer.units
            total_flops += flops

        elif layer_type == 'MaxPooling2D' or layer_type == 'AveragePooling2D':
            pool_height, pool_width = layer.pool_size
            stride_height, stride_width = layer.strides
            input_shape = layer.input_shape[1:]
            if stride_height is None:
                stride_height = pool_height
            if stride_width is None:
                stride_width = pool_width
            flops = (input_shape[0] / stride_height) * input_shape[1] * input_shape[2]
            total_flops += flops

    return total_flops

def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', help='Enter Model Dirrectory')
    opts = parser.parse_args()
    return opts

if __name__ == '__main__':
    opts = parseArgs(sys.argv)
    model = load_model_from_h5(os.path.join(opts.model_dir,'model','best_model.h5'))
    model.summary()
    flops = calculate_flops(model)
    print("Total Number of FLOPS: ",((flops/1000000),2),' Million')


    # This code was developed and designed by Mabon Ninan
    # UCDasec