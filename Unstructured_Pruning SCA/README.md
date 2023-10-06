# Unstructured Pruning
Unstructured pruning is also called magnitude pruning. Unstructured pruning converts some of the parameters or weights with smaller magnitude into zeros.

Dense: lots of non-zero values
Sparse: lots of zeros

Unstructured pruning converts an original dense network into a sparse network. The size of the parameter matrix (or weight matrix) of the sparse network is the same as the size of parameter matrix of the original network. Sparse network has more zeros in their parameter matrix.
The unstructured pruning does not consider any relationship between the pruned weights


![image](https://github.com/UCdasec/TinyPower/assets/54579704/3d3aeefb-403d-443d-8fe7-8c47a6afde21)


# Pruning of Deep Learning Model for HW/ID Leakage Detection
This repository contains a code for training a pruned deep learning model for HW/ID leakage detection. The code is written in Python and uses the TensorFlow library.

# Requirements:

 *  Python 3.6 or higher
  
 *  TensorFlow 2.0 or higher
  
 *  NumPy
  
# Usage:

To train a pruned deep learning model for HW/ID leakage detection, run the following command:
```python 
python train.py --input <input_data_file> --output <output_directory> --model_dir <model_directory> --network_type <network_type> --target_byte <target_byte> --pruning_rate <pruning_rate>
```
# Arguments:

* input: The path to the input data file. The input data file should be in a NumPy .npy format.

* output: The path to the output directory. The output directory will contain the trained model and the training logs.

* model_dir: The path to the model directory. The model directory will contain the pre-trained model that will be used as the starting point for pruning.

* network_type: The type of network to train. The supported network types are hw_model and ID.

* target_byte: The target byte to detect leakage for. The target byte should be an integer in the range 0-255.

* pruning_rate: The percentage of weights to prune. The pruning rate should be a float in the range 0.0-1.0.
