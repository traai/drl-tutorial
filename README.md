# Tutorial Series on Deep Reinforcement Learning
Code accompanying a lecture series on Deep Reinforcement Learning at NTNU: https://www.ntnu.edu/web/ailab/dl_tutorial

![](https://github.com/traai/drl-tutorial/blob/master/assets/drl-tutorial.png?raw=true)

## Setting up your machine to run/play with code in this repo

1. Get __Python 3.6__ version of [__Anaconda__](https://www.anaconda.com/download/)

### On a Mac/(Linux?):
2. Run the following commands to set up Python environment
	* `conda create --name <envname> python=3`
	* `source activate <envname>`
	* `conda install matplotlib`
	* `pip install gym`
	* `pip install --upgrade tensorflow`
	* `pip install keras`
	* `pip install h5py`

### On Win:
2. Run the following commands to set up Python environment
	* `conda create --name <envname> python=3`
	* `activate <envname>`
	* `conda install matplotlib`
	* `pip install gym`
	* `pip install --upgrade tensorflow`
	* `pip install keras`
	* `pip install h5py`

---

# Tasks

## [Value based methods](https://github.com/traai/drl-tutorial/tree/master/value)

![](https://github.com/traai/drl-tutorial/blob/master/assets/catch.gif?raw=true)

1. Try **another RL environment/problem** with the DQN implementation in `dqn.py` e.g. by changing size of state space for the **catch** problem, or from [OpenAI Gym](https://gym.openai.com/envs/) 

2. Play with the **value network architecture** (e.g. add or reduce layers/layer sizes)

3. Note that the **basic implementation has only one network**. Let's call it the online network. It is used to collect data as well as to compute targets in the `compute_targets` function. For easy problems, this can be fine. But for Atari, this will cause the computed targets to move in detrimental ways. Can you **implement a second network**, which you use **to compute targets**. It should be a clone of the online net (same architecture). Let's call it the target network. Parameters of online and target network should be **synced** at regular intervals ~ target network will be **frozen** between intervals. [Here (slide 6)](https://drive.google.com/file/d/0BxXI_RttTZAhVUhpbDhiSUFFNjg/view) is some intuition on this -- accompanying [video](https://www.youtube.com/watch?v=fevMOp5TDQs). **Hint**: Use Keras functions `set_weights()` and `get_weights()`. For example, the following could be set up:

	```python
	target_network.set_weights(online_network.get_weights())
	```
More hints [here](https://github.com/keon/deep-q-learning/blob/master/ddqn.py), and should also help with task 6!

### Extra challenges:

4. Try [prioritised sampling](https://arxiv.org/pdf/1511.05952.pdf) from the replay buffer. **Hint**: You can try to plug in the prioritised experience replay [code](https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py) from OpenAI baselines.

5. Try a [dueling network architecture](https://arxiv.org/pdf/1511.06581.pdf). Example implementation for how to change the network architecture to make it dueling can be found [here](https://github.com/matthiasplappert/keras-rl/blob/master/rl/agents/dqn.py#L89).

6. Try **selecting** assumed optimal (argmax) action using online net and **evaluating** it using target net to compute targets. This creates an ensembling effect and makes target estimates better. It is also the idea behind [double DQN](https://arxiv.org/pdf/1509.06461.pdf).


### Very extra challenge:

7. Try putting prioritised sampling, dueling architecture, and double DQN learning together!

## [Policy based methods](https://github.com/traai/drl-tutorial/tree/master/pg)

![](https://github.com/traai/drl-tutorial/blob/master/assets/balance.gif?raw=true)

1. Try **another RL environment/problem** e.g. from [OpenAI Gym](https://gym.openai.com/envs/), with the simple policy gradient algorithm implementation in `pg.py`.

2. Play with the **policy network architecture**.

3. Try returns **with and without discounting**. Do gradients become more noisy without discounting? Does it take longer to train to get the same performance as with discounting?

### Extra challenges:

4. Full episode returns are called Monte Carlo returns. These have high variance. Discounting helps to some extent. But, gradients are still noisy, since returns modulate the gradients. Try including a baseline to reduce the variance in episodic returns. This can be done (**already done for you in** `pg_with_baseline_task.py`**!**) by setting up another network that outputs the value (expected return) of a state. Can you use it to **compute the advantages**, as opposed to full returns. Then you can you the advantages to modulate the gradients! Solution in `pg_with_baseline.py`, but try figuring it out yourself first to see if you get the concept of baselines and action advantages.

5. Try using same network for policy and value/baseline. **Hint**: Last but one layer (before output) can have **two heads**, one giving the policy and other giving value. A single loss function (summed) for both can also be constructed, to compute gradients more efficiently. 

### Very extra challenge:
6. Try changing the loss/objective function to make policy updates [proximal](https://arxiv.org/pdf/1707.06347.pdf). Proximality here means that the updates to the policy network should be such that the updated policy does not become very different from the policy before the update.

## Actor-critic
1. Try including **bootstrapping** (as opposed to Monte Carlo sampling) in returns for each step during each episode. You may **use the second network/value function** to carry out the bootstrapping. In doing so, this network plays the role of a **critic**.

---

# Un-rules

1. Consider this as an **open book challenge**. Work your way through the tasks in any order you like. 

2. Team up with your neighbour or work through these by yourself. If something is not clear, ask me/Slack.

3. Basic **implementations** of value and policy based methods, and the **environment setup instructions** are provided. Go through these and see if you understand everything well. If not, ask. 

4. Either **build on top** of these implementations as you go through the tasks, or **implement your own from scratch!** 

5. Refer online lectures, blog posts, available code etc. Or ask.

	> **Best practices from John Schulman (OpenAI)** when working with deep RL
		[Video](https://www.youtube.com/watch?v=8EcdaCk9KaQ&feature=youtu.be)
		[Slides](https://drive.google.com/file/d/0BxXI_RttTZAhc2ZsblNvUHhGZDA/view) [Notes](https://github.com/williamFalcon/DeepRLHacks)

	> [More best practices from OpenAI](https://blog.openai.com/openai-baselines-dqn/)
 	
 	> DQN [intuition](https://www.youtube.com/watch?v=fevMOp5TDQs) from **Vlad Mnih (Deepmind)**
	
	> [Policy gradients](https://www.youtube.com/watch?v=tqrcjHuNdmQ) [intuition](http://karpathy.github.io/2016/05/31/rl/) from **Andrej Karpathy (Tesla)**
	
	> [Some online tutorials](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0) with code samples
	
	> [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures) -- **must watch if you want to start working in the field!**
	
	> Full [UCL course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) on RL by **David Silver (Deepmind)**
	
	> Full [UC Berkeley course](http://rll.berkeley.edu/deeprlcourse/) on **Deep RL** 

6. **Use Slack during/after lecture** to discuss issues and share thoughts on your implementations.

7. `Have fun!`