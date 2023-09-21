# CNN Based Method
This folder contains the code of CNN based Side-Channel Attack method. In this file, it explains what function that each source code file have and what envrionment that the code can run on, and also give detailed the command line of train and test the CNN based side-channel attack model.

## Content

## Dataset
All the dataset can be accessed by the follow link
<wait to be add when the dataset is released>

## Requirements
This project is developed with Python3.6, Tensorflow 2.3 on a Ubuntu 18.04 OS

## Usage
Below is the command line for the train and test the CNN based attack method, when you run, you need to replace the pseudo parameters with your own path or specifications
* Train:

    python train.py --input path_to_dataset --output path_to_save_the_model
                    --verbose --target_byte TARGET_BYTE
                    --network_type choose_network_type{hw_model,mlp,cnn2,wang,cnn}
                    --attack_window ATTACK_WINDOW

* Test:

    python test.py --input path_to_dataset --output path_to_save_the_test_results
                   --model_file MODEL_FILE --verbose --target_byte TARGET_BYTE
                   --network_type choose_network_type{wang,cnn2,cnn,mlp}
                   --attack_window ATTACK_WINDOW

### Usage Examples
For example, if you have the following specification:



