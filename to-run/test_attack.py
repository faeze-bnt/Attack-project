## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import random
import tensorflow as tf
import numpy as np
import time

# from setup_mnist import MNIST, MNISTModel

# from l2_attack import CarliniL2

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


def generate_data(data, samples, targeted=True, start=0, inception=False):
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
                seq = range(data[3].shape[1])

            for j in seq:
                if (j == np.argmax(data[3][start+i])) and (inception == False):
                    continue
                inputs.append(data[2][start+i])
                targets.append(np.eye(data[3].shape[1])[j])
        else:
            inputs.append(data[2][start+i])
            targets.append(data[3][start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


# if __name__ == "__main__":


#     with tf.compat.v1.Session() as sess:

#         data, model =  MNIST(), MNISTModel("my_LeNet5_best.h5", sess)

#         attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)

#         # print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
#         # print(np.size(data))
#         # print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")

#         inputs, targets = generate_data(data, samples=1, targeted=True,
#                                         start=0, inception=False)
#         timestart = time.time()
#         # print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
#         # print((inputs.shape))
#         # print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        
#         adv = attack.attack(inputs, targets)
#         timeend = time.time()
        
#         print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

#         print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
#         print(len(adv))
#         print(inputs)
#         print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")

#         for i in range(len(adv)):
#             print("Valid:")
#             #show(inputs[i])
#             print("Adversarial:")
#             #show(adv[i])
            
#             print("Classification:", model.model.predict(adv[i:i+1]))

#             print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)
