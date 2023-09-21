#!/usr/bin/python3
import os
import pdb
import h5py
import random
import tensorflow as tf
import numpy as np
from collections import defaultdict
import ast
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

import loadData


Sbox = [99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71,
        240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216,
        49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160,
        82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208,
        239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188,
        182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96,
        129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211,
        172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186,
        120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97,
        53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140,
        161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22]


HW_byte = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2,
           3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3,
           3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3,
           4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
           3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
           6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
           4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5,
           6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]


def ranking_curve(preds, key, plaintext, target_byte, rank_root, leakage_model='HW', trace_num_max=500):
    """
    - preds : the probability for each class (n*256 for a byte, n*9 for Hamming weight)
    - real_key : the key of the target device
    - device_id : id of the target device
    - model_flag : a string for naming GE result
    # max trace num for attack
    """
    # GE/SR is averaged over 100 attacks 
    num_averaged = 100
    guessing_entropy = np.zeros((num_averaged, trace_num_max))
    success_flag = np.zeros((num_averaged, trace_num_max))

    real_key = key[target_byte]
    plaintext = plaintext[:, target_byte]

    # attack multiples times for average
    for time in range(num_averaged):
        # select the attack traces randomly
        random_index = list(range(plaintext.shape[0]))

        #         ## customized by HL
        #         print(f"random_index shape {len(random_index)}, max value {max(random_index)}, min value {min(random_index)}")

        random.shuffle(random_index)
        random_index = random_index[0:trace_num_max]

        #         ## customized by HL
        #         print(f"random_index shape after slicing {len(random_index)}, max value {max(random_index)}, min value {min(random_index)}")

        # initialize score matrix
        score_mat = np.zeros((trace_num_max, 256))
        for key_guess in range(0, 256):
            for i in range(0, trace_num_max):
                initialState = int(plaintext[random_index[i]]) ^ key_guess
                sout = Sbox[initialState]
                if leakage_model == 'ID':
                    label = sout
                elif leakage_model == 'HW':
                    label = HW_byte[sout]
                try:
                    score_mat[i, key_guess] = preds[random_index[i], label]
                except Exception as e:
                    pdb.set_trace()
                    print(e.message)
        score_mat = np.log(score_mat + 1e-40)

        #         ## customized by HL
        #         print(f"score_mat {score_mat}")

        for i in range(0, trace_num_max):
            log_likelihood = np.sum(score_mat[0:i+1, :], axis=0)
            ranked = np.argsort(log_likelihood)[::-1]
            guessing_entropy[time, i] = list(ranked).index(real_key)
            if list(ranked).index(real_key) == 0:
                success_flag[time, i] = 1

    guessing_entropy = np.mean(guessing_entropy, axis=0)

    # define the saving path
    os.makedirs(rank_root, exist_ok=True)

    # only plot guess entry
    plt.figure(figsize=(8, 6))
    plt.plot(guessing_entropy[0:trace_num_max], color='red')
    plt.title('Leakage model: {}, target byte: {}'.format(leakage_model, target_byte))
    plt.xlabel('Number of trace')
    plt.ylabel('Key Rank')
    fig_save_path = os.path.join(rank_root, 'ranking_curve.png')
    plt.savefig(fig_save_path)
    plt.show()
    print('[LOG] -- ranking curve save to path: ', fig_save_path)

    # saving the ranking raw data
    raw_save_path = os.path.join(rank_root, 'ranking_raw_data.npz')
    x = list(range(len(guessing_entropy)))
    np.savez(raw_save_path, x=x, y=guessing_entropy)
    print('[LOG] -- ranking raw data save to path: ', raw_save_path)


def get_the_labels(textins, key, target_byte):
    labels = []
    for i in range(textins.shape[0]):
        text_i = textins[i]
        label = loadData.aes_internal(text_i[target_byte], key[target_byte])
        labels.append(label)

    labels = np.array(labels)
    return labels

