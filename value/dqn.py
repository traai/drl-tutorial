from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from catch import *

class ExperienceReplay(object):
    def __init__(self, max_memory=500):
        self.max_memory = max_memory
        self.memory = list()

    def add(self, sars_d):
        # memory[i] = [state_t, action_t, reward_t, state_t+1, game_over?]
        self.memory.append(sars_d)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def sample_batch(self, batch_size=50):
        len_memory = len(self.memory)
        return_batch_size = min(len_memory, batch_size)
        batch = []
        for i, idx in enumerate(np.random.randint(0, len_memory, size=return_batch_size)):
            batch.append(self.memory[idx])
        return batch


def compute_targets(model, batch, discount=0.9):
    # NB. For each experience tuple in the batch, only the action in the 
    # experience tuple influences the updates to the network.
    # So, we set targets of action values for the actions not taken/not in tuple 
    # to old (before update) Q values of those actions -- Line 49
    # This would make the "mse" loss contribution from those outputs zero. The 
    # chain rule of gradient computations would make gradients w.r.t weights 
    # along paths in the network leading to these output action values zero.
    batch_size = len(batch)
    state_dim = batch[0][0].shape[1]
    
    num_actions = model.output_shape[-1]
    
    inputs = np.zeros((batch_size, state_dim))
    targets = np.zeros((batch_size, num_actions))
    
    for i, sars_d in enumerate(batch):
        state_t, action_t, reward_t, state_tp1, game_over = sars_d

        inputs[i] = state_t
        targets[i] = model.predict(state_t)[0]

        Q_next_sa = np.max(model.predict(state_tp1)[0])
        if game_over:
            # Since no next state, Bellman target only involves immediate reward
            targets[i, action_t] = reward_t 
        else:
            # Full Bellman target: reward_t + gamma * max_a' Q(s', a')
            targets[i, action_t] = reward_t + discount * Q_next_sa
    return inputs, targets

if __name__ == "__main__":
    # Parameters
    epsilon = .1  # Exploration
    num_actions = 3  # [move_left, stay, move_right]
    epoch = 1000 # Number of episodes in this case
    max_memory = 500 # Max replay memory size
    hidden_size = 100 # Size of the hidden layers. NB. Can of course have different number of neurons per layer.
    batch_size = 50
    grid_size = 10 # Increase to make problem harder
    discount = 0.9 # gamma in Bellman equations/discounted return computations

    # Building a feed forward network with two hidden layers in Keras.
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    # Define loss function
    model.compile(sgd(lr=.2), "mse")

    # If you want to continue training from a previously learnt model, just 
    # uncomment the line bellow.
    # model.load_weights("model.h5")

    # Instantiate the problem/environment/game
    env = Catch(grid_size)

    # Initialize experience replay buffer
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0
    for e in range(epoch):
        loss = 0.0
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t
            # Choose action with epsilon greedy exploration policy
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # Act, and get immediate reward, new state, and whether game's over
            input_t, reward, game_over = env.act(action)
            
            # Just bookeeing
            if reward == 1:
                win_cnt += 1

            # Store experience in replay buffer
            exp_replay.add([input_tm1, action, reward, input_t, game_over])

            # Get minibatch of experiences from replay buffer to train on
            batch = exp_replay.sample_batch(batch_size=batch_size)

            # Compute targets given the batch
            inputs, targets = compute_targets(model, batch, discount)

            loss += model.train_on_batch(inputs, targets)
        print("Epoch {:03d}/{} | Loss {:.4f} | Win count {}".format(e, epoch-1, loss, win_cnt))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
