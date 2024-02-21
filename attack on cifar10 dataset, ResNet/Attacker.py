
import tensorflow as tf
import random
import tensorflow as tf
import numpy as np
import time
import pickle

from l2_attack          import CarliniL2
from resnet20     		import ResNet20



def generate_data(x_test, y_test, samples, targeted=True, start=0, inception=False):
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
                seq = range(y_test.shape[1])

            for j in seq:
                if (j == np.argmax(y_test[start+i])) and (inception == False):
                    continue
                inputs.append(x_test[start+i])
                targets.append(np.eye(y_test.shape[1])[j])
        else:
            inputs.append(x_test[start+i])
            targets.append(y_test[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets



def MainAttack(model_type, x_test, y_test, input_shape, test_samples):
    
    with tf.compat.v1.Session() as sess:

        #--------------------------------------------------------------------------------
        # Call a model 
        feke_model = ResNet20(input_shape=input_shape, depth=None, pre_softmax=True)

        model_name = 'cifar10_%s_model.h5' % model_type
        feke_model.model.load_weights(model_name)
        print("Weigths are loaded from disk")

        attacker = CarliniL2(sess, feke_model, input_shape, batch_size=9, max_iterations=1000, confidence=0)
        inputs, targets = generate_data(x_test, y_test, samples=test_samples, targeted=True, 
                                        start=0, inception=False)
        
        timestart = time.time()
        adv = attacker.attack(inputs, targets)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run ",len(inputs),"samples.")
        print("Total number of adv samples are: ", len(adv))
        
        ##----- save the attack outputs in files for later use
        with open('adv_attack_x.pkl', 'wb') as file:
            pickle.dump(adv, file)
        with open('adv_attack_y.pkl', 'wb') as file:
            pickle.dump(targets, file)

    return adv, targets

