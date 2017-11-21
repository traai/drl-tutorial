from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
from catch import *
import time

plt.ion()

if __name__ == "__main__":
    # Make sure this grid size matches the value used for training
    grid_size = 10

    with open("model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("model.h5")
    model.compile("sgd", "mse")

    # Define environment, game
    env = Catch(grid_size)
    c = 0
    for e in range(5):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        plt.imshow(input_t.reshape((grid_size,)*2),
                   interpolation='none', cmap='gray')
        plt.draw()
        plt.pause(.001)

        c += 1
        while not game_over:
            # time.sleep(.5)
            input_tm1 = input_t

            # get next action
            q = model.predict(input_tm1)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)

            plt.imshow(input_t.reshape((grid_size,)*2),
                       interpolation='none', cmap='gray')
            plt.draw()
            plt.pause(.001)

            c += 1
