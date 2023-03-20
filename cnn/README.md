# CNN Based Method
This folder contains the code of CNN based Side-Channel Attack method. In this file, it explains what function that each source code file have and what envrionment that the code can run on, and also give detailed the command line of train and test the CNN based side-channel attack model.

## Content
This instructions for each python source code file is list below:
* checking\_tool.py: This is the source code including some support functions for the CNN based attack method.
* finetune.py: This is the source code for fine-tuning the trained cnn model
* model\_zoo.py: This is the source code for our cnn model definition
* process\_data.py: This is the source code for process and prepare the dataset
* test.py: This is the source code for testing cnn based side-channel attack model
* train.py: This is the source code for training cnn based side-channel attack model

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
* training data is stored at ./dataset/xmega\_unmasked/PC1\_CB1\_TDX1\_K0\_U\_200k.npz
* testing data is stored at ./dataset/xmega\_unmasked/PC1\_CB1\_TDX1\_K1\_U\_20k.npz
* store your trained model at ./trained\_model/unmasked\_xmega\_cnn.h5
* store your test results at ./test\_results/unmasked\_xmega\_cnn
* choose the cnn2 network (the one same as in the ASCAD paper)
* choose attack window [1800, 2800]
* choose target byte 0

Then you should run the following commands

* Train:

    python train.py --input ./dataset/xmega\_unmasked/PC1\_CB1\_TDX1\_K0\_U\_200k.npz
                    --output ./trained\_model/unmasked\_xmega\_cnn.h5
                    --verbose
                    --target_byte 0
                    --network_type cnn2
                    --attack_window 1800\_2800

* Test:

    python test.py --input ./dataset/xmega\_unmasked/PC1\_CB1\_TDX1\_K1\_U\_20k.npz
                   --output ./test\_results/unmasked\_xmega\_cnn
                   --model_file ./trained\_model/unmasked\_xmega\_cnn.h5
                   --verbose
                   --target_byte 0
                   --network_type cnn2
                   --attack_window 1800\_2800


## Contacts
Chenggang Wang: wang2c9@mail.uc.edu, University of Cincinnati

Boyang Wang: boyang.wang@uc.edu, University of Cincinnati
