## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import random
import tensorflow as tf
import numpy as np


# Enable XLA devices
tf.config.optimizer.set_jit(True)

def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def generate_data(test_data, test_labels, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(test_data[start+i])
                targets.append(np.eye(test_labels.shape[1])[j])

            
        else:
            inputs.append(test_data[start+i])
            targets.append(test_labels[start+i])
        print(seq)
        print(start+i)
        print(test_labels[start+i])
        print(len(inputs))

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets

