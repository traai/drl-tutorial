# Tasks

## Value based methods

1. Try **another environment/task** e.g. by changing size of state space for the catching problem, or from [OpenAI Gym](https://gym.openai.com/envs/) 

2. Play with **value network architecture** (e.g. add or reduce layers/layer sizes)

3. Note that the **basic implementation has only one network**. Let's call it the online net. It is used to collect data as well as to compute targets in the `compute_targets` function. For easy problems, this can be fine. But for Atari, this will cause the computed targets to move in detrimental ways. Can you **implement a second network**, which you use **to compute targets**. It should be a clone of the online net (same architecture). Let's call it the target network. Parameters of online and target network should be **synced** at regular intervals ~ target network will be **frozen** between intervals. Intuition on DQN: [https://drive.google.com/file/d/0BxXI_RttTZAhVUhpbDhiSUFFNjg/view]()

### Extra challenges:

4. Try **prioritised sampling** from the replay buffer: [https://arxiv.org/pdf/1511.05952.pdf](). **Hint**: You can try to plug in the [prioritised experience replay code from OpenAI baselines](https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py).

5. Try a **dueling network architecture**: [https://arxiv.org/pdf/1511.06581.pdf](). Example implementation for how to change the network architecture to make it dueling can be found [here](https://github.com/matthiasplappert/keras-rl/blob/master/rl/agents/dqn.py#L89).

6. Try selecting assumed optimal (argmax) action using online net and evaluating it using target net to compute targets:[ https://arxiv.org/pdf/1509.06461.pdf]() (**double DQN**)

### Very extra challenge:

* Try putting prioritised sampling, dueling architecture, and double Q learning together!

## Policy based methods
1. Try **another environment/task** e.g. from [OpenAI Gym](https://gym.openai.com/envs/)

2. Play with **policy network architecture**.

3. Try returns **with and without discounting**. Do gradients become more noisy without discounting? Does it take longer to train to get the same performance as with discounting?

4. Full episode returns are called Monte Carlo returns. These have high variance. Discounting helps to some extent. But, gradients are still noisy, since returns modulate the gradients. Try including a baseline to reduce the variance in episodic returns. This can be done (**already done for you in** `pg_with_baseline.py`**!**) by setting up another network that outputs the value (expected return) of a state. Can you use it to **compute the advantages**, as opposed to full returns. Then you can you the advantages to modulate the gradients!

5. Try using same network for policy and value/baseline. **Hint**: Last but one layer (before output) can have **two heads**, one giving the policy and other giving value. A single loss function (summed) for both can also be constructed, to compute gradients more efficiently. 

### Very extra challenge:
* Try changing the loss/objective function to make policy updates **proximal**: [https://arxiv.org/pdf/1707.06347.pdf](). Proximality here means that the updates to the network should change the policy such that the updated policy is not very different from this policy.

## Actor-critic
1. Try including **bootstrapping** (as opposed to Monte Carlo sampling) in returns for each step during each episode. You may **use the second network/value function** to carry out the bootstrapping. In doing so, this network plays the role of a **critic**.

---

# Un-rules

1. Consider this as an **open book challenge**. Work your way through the tasks in any order you like. 

2. Team up with your neighbour or by yourself. If something is not clear, ask me.

3. Basic **implementations** of value and policy based methods, and the **environment setup instructions** are provided. Go through these and see if you understand everything well. If not, ask. 

4. Either **build on top** of these implementations as you go through the tasks, or **implement your own from scratch!** 

5. Refer online lectures, blog posts, available code etc. Or ask.

	* **Best practices** when working with deep RL
		* Video: [https://www.youtube.com/watch?v=8EcdaCk9KaQ&feature=youtu.be]()
		* Slides: [https://drive.google.com/file/d/0BxXI_RttTZAhc2ZsblNvUHhGZDA/view]()
		* Quick notes: [https://github.com/williamFalcon/DeepRLHacks]()

	* More best practices: [https://blog.openai.com/openai-baselines-dqn/]()
 	* DQN intuition: [https://www.youtube.com/watch?v=fevMOp5TDQs]()
	* Policy gradients intuition: [https://www.youtube.com/watch?v=tqrcjHuNdmQ](), [http://karpathy.github.io/2016/05/31/rl/]()
	* Some online tutorials with code samples: [https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0]()
	* **Deep RL Bootcamp** **-- must watch if you want to start working in the field!**: [https://sites.google.com/view/deep-rl-bootcamp/lectures]()

6. **Use Slack during/after lecture** to discuss issues and share thoughts on your implementations.

7. `Have fun!`