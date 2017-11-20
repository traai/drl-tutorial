"""
Simple policy gradient with baseline correction (advantage estimates) in Keras

"""
import gym
import numpy as np
import time

from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers


class Agent(object):

    def __init__(self, input_dim, output_dim, policy_hidden_dims=[32, 32], baseline_hidden_dims=[]):
        """Gym Playing Agent

        Args:
            input_dim (int): the dimension of state.
                Same as `env.observation_space.shape[0]`

            output_dim (int): the number of discrete actions
                Same as `env.action_space.n`

            hidden_dims (list): hidden dimensions

        Methods:

            private:
                __build_graph -> None
                    It creates a base model and the training operations
                    Base model outputs action probability
                    Training operations extend compute graph with the loss/
                    objective function, and the gradient computation opertion

            public:
                get_action(state) -> action
                fit(state, action, reward) -> None
        """

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.__build_policy_graph(input_dim, output_dim, policy_hidden_dims)
        self.__build_baseline_graph(input_dim, baseline_hidden_dims)

    def __build_policy_graph(self, input_dim, output_dim, hidden_dims=[32, 32]):
        # Create a policy network
        # Info on how to implement own loss function: 
        #     https://github.com/fchollet/keras/issues/2662
        
        self.X = layers.Input(shape=(input_dim,))
        net = self.X
        for h_dim in hidden_dims:
            net = layers.Dense(h_dim)(net)
            net = layers.Activation("relu")(net)
        net = layers.Dense(output_dim)(net)
        
        # Softmax outout/policy
        net = layers.Activation("softmax")(net)

        self.policy_model = Model(inputs=self.X, outputs=net)

        policy_out = self.policy_model.output
        action_onehot_placeholder = K.placeholder(shape=(None, self.output_dim), 
                                                    name="action_onehot")
        advantage_placeholder = K.placeholder(shape=(None,),
                                                    name="advantage")

        action_prob = K.sum(policy_out * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)

        # Loss = - Objective = - Mean across sample actions along trajectory
        loss = - log_action_prob * advantage_placeholder
        loss = K.mean(loss)

        # Can use different types of gradient descent methods. - try it!
        adam = optimizers.Adam()

        updates = adam.get_updates(params=self.policy_model.trainable_weights,
                                   loss=loss)

        self.policy_train_fn = K.function(inputs=[self.policy_model.input,
                                                  action_onehot_placeholder,
                                                  advantage_placeholder],
                                          outputs=[],
                                          updates=updates)

        
    def __build_baseline_graph(self, input_dim, hidden_dims=[]):
        # Create a baseline network and its training operations
        self.X = layers.Input(shape=(input_dim,))
        net = self.X
        for h_dim in hidden_dims:
            net = layers.Dense(h_dim)(net)
            net = layers.Activation("relu")(net)
        net = layers.Dense(1)(net)        
        self.baseline_model = Model(inputs=self.X, outputs=net)

        baseline_out = self.baseline_model.output

        discounted_return_placeholder = K.placeholder(shape=(None,),
                                                    name="discounted_return")

        baseline_loss = K.mean(K.square(baseline_out - discounted_return_placeholder), axis=-1)
        
        adam = optimizers.Adam()
        baseline_updates = adam.get_updates(params=self.baseline_model.trainable_weights,
                                            loss=baseline_loss)
        self.baseline_train_fn = K.function(inputs=[self.baseline_model.input,
                                                    discounted_return_placeholder],
                                            outputs=[],
                                            updates=baseline_updates)        


    def get_action(self, state):
        """Returns an action at given `state`

        Args:
            state (1-D or 2-D Array): It can be either 1-D array of shape (state_dimension, )
                or 2-D array shape of (n_samples, state_dimension)

        Returns:
            action: an integer action value ranging from 0 to (n_actions - 1)
        """
        shape = state.shape

        if len(shape) == 1:
            assert shape == (self.input_dim,), "{} != {}".format(shape, self.input_dim)
            state = np.expand_dims(state, axis=0)

        elif len(shape) == 2:
            assert shape[1] == (self.input_dim), "{} != {}".format(shape, self.input_dim)

        else:
            raise TypeError("Wrong state shape is given: {}".format(state.shape))

        action_prob = np.squeeze(self.policy_model.predict(state))
        assert len(action_prob) == self.output_dim, "{} != {}".format(len(action_prob), self.output_dim)
        return np.random.choice(np.arange(self.output_dim), p=action_prob)

    def compute_discounted_R(self, R, discount_rate=.99):
        """Returns discounted rewards
    
        Args:
            R (1-D array): a list of `reward` at each time step
            discount_rate (float): Will discount the future value by this rate
    
        Returns:
            discounted_r (1-D array): same shape as input `R`
                but the values are discounted
    
        Examples:
            >>> R = [1, 1, 1]
            >>> compute_discounted_R(R, .99) # before normalization
            [1 + 0.99 + 0.99**2, 1 + 0.99, 1]
        """
        discounted_R = np.zeros_like(R, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(R))):
    
            running_add = running_add * discount_rate + R[t]
            discounted_R[t] = running_add
    
        # Making returns mean 0.0 and STD 1.0
        # This keeps gradients from blowing up without changing the relative 
        # importance associated with each action along a trajectory that modulates 
        # its future probability of being chosen. 
        # This is not the same as having a baseline.
        discounted_R -= discounted_R.mean() / discounted_R.std()
    
        return discounted_R

    def fit(self, S, A, R):
        """Train a network

        Args:
            S (2-D Array): `state` array of shape (n_samples, state_dimension)
            A (1-D Array): `action` array of shape (n_samples,)
                It's simply a list of int that stores which actions the agent chose
            R (1-D Array): `reward` array of shape (n_samples,)
                A reward is given after each action.

        """
        action_onehot = np_utils.to_categorical(A, num_classes=self.output_dim)
        discounted_return = self.compute_discounted_R(R)
        value = np.squeeze(self.baseline_model.predict(S))
        advantage = discounted_return - value
        assert S.shape[1] == self.input_dim, "{} != {}".format(S.shape[1], self.input_dim)
        assert action_onehot.shape[0] == S.shape[0], "{} != {}".format(action_onehot.shape[0], S.shape[0])
        assert action_onehot.shape[1] == self.output_dim, "{} != {}".format(action_onehot.shape[1], self.output_dim)
        assert len(discounted_return.shape) == 1, "{} != 1".format(len(discounted_return.shape))
        assert len(advantage.shape) == 1, "{} != 1".format(len(advantage.shape))

        self.baseline_train_fn([S, discounted_return])
        self.policy_train_fn([S, action_onehot, advantage])

def train_on_episode(env, agent):
    """Returns an episode reward

    (1) Play until the game is done
    (2) The agent will choose an action according to the policy
    (3) When it's done, it will train from the game play

    Args:
        env (gym.env): Gym environment
        agent (Agent): Game Playing Agent

    Returns:
        total_reward (int): total reward earned during the whole episode
    """
    done = False
    S = []
    A = []
    R = []

    s = env.reset()
    total_reward = 0
    while not done:
        a = agent.get_action(s)
        s_next, r, done, info = env.step(a)
        total_reward += r

        S.append(s)
        A.append(a)
        R.append(r)

        s = s_next

        if done:
            S = np.array(S)
            A = np.array(A)
            R = np.array(R)

            agent.fit(S, A, R)

    return total_reward


def main():
    try:
        env = gym.make("CartPole-v0")
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        agent = Agent(input_dim, output_dim, [32, 32], [])

        for episode in range(1000):
            reward = train_on_episode(env, agent)
            print(episode, reward)

        for test_episode in range(5):
            done = False
            s = env.reset()
            total_reward = 0
            while not done:
                env.render()
                s_next, r, done, info = env.step(agent.get_action(s))
                total_reward += r
                s = s_next
            print("Total reward in test episode {}: {}".format(test_episode + 1, total_reward))
    finally:
        env.close()

if __name__ == '__main__':
    main()