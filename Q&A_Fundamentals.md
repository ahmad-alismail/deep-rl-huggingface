
# Unit 1
1.  What is the goal of any reinforcement learning agent?
2.  What is the RL process?
3.  Why do we discount rewards in RL?
4.  How do we find the optimal policy in RL?
5.  How does deep learning come into play with RL?
6.  Can you explain the difference between policy-based and value-based methods in RL?
7.  How does the expected cumulative reward relate to the reward hypothesis in RL?
8.  How does the agent learn from the environment in RL?
9.  what is the function of agent's policy and why it called as agent's 'brain'?
10.  How does neural network estimate the action to take or the value of a state in Deep RL?
11. What is the difference Policies and Value Functions?
12. Why are value functions useful?

##### 1.  What is the goal of any reinforcement learning agent?
 * The goal of any reinforcement learning agent is to maximize its expected cumulative reward (also called expected return) because RL is based on the reward hypothesis, which is that all goals can be described as the maximization of the expected cumulative reward.
 
##### 2.  What is the RL process?
* The RL process is a loop that outputs a sequence of state, action, reward and next state.

##### 3.  Why do we discount rewards in RL?
* In RL, rewards that come sooner (at the beginning of the game) are more probable to happen since they are more predictable than the long term future reward. To account for this, we discount the rewards so that more immediate rewards are given more weight.

##### 4.  How do we find the optimal policy in RL?
* To find the optimal policy in RL, there are two methods: By training your policy directly (policy-based methods) and by training a value function that tells us the expected return the agent will get at each state and use this function to define our policy (value-based methods)

##### 5.  How does deep learning come into play with RL?
* Deep RL refers to using deep neural networks to estimate the action to take (policy-based) or to estimate the value of a state (value-based)

##### 6.  Can you explain the difference between policy-based and value-based methods in RL?
  * Policy-based methods in RL train the agent's policy directly, whereas value-based methods train a value function that is used to define the policy.
OR
-   _Value-based reinforcement learning method_: learning an action-value function that will tell us what's the **most valuable action to take given a state and action**.
-   _Policy-based reinforcement learning method_: learning a policy that will **gives us a probability distribution over actions**.

##### 7.  How does the expected cumulative reward relate to the reward hypothesis in RL?
* The expected cumulative reward is based on the reward hypothesis, which states that all goals in RL can be described as the maximization of the expected cumulative reward.

##### 8.  How does the agent learn from the environment in RL?
 * The agent learns from the environment in RL by interacting with it through trial and error and receiving rewards (negative or positive) as feedback.

##### 9.  What is the function of agent's policy and why it called as agent's 'brain'?
* The agent's policy is its "brain," which tells us what action to take given a state. The optimal policy is the one that maximizes the expected return.

##### 10.  How does neural network estimate the action to take or the value of a state in Deep RL?
 * Neural networks in Deep RL estimate the action to take or the value of a state by learning from input data, such as the current state of the environment. Through training, the network learns to predict the optimal action or value given a particular input state.

##### 11. What is the difference Policies and Value Functions?
* A **policy** is a learned mapping from states to actions. 
> Given a state, what action to take now?

* A **value function** is a learned mapping from states to values. 
> Given a state, what is the expected discounted return the agent can get if it starts in that state, and then act accrding to our policy? (i.e., our policy is going to the state with the highest value)

##### 12. Why are value functions useful?
*  Because you can infer an optimal policy from an optimal q-value function
> Value functions can also map pairs of (state,action) to values. In this case, they are called q-value functions.
##### 13. Why discount factor is useful?
* Bird in the hand better than 10 on the tree. The future reward is not so important as immediate reward. The discount factor is an incentive to take the **shortest** path to the reward.

# Unit 2 
##### 1.  What are the two main types of RL methods?

-   Policy-based methods and value-based methods.

##### 2.  How does the Bellman equation simplify the process of calculating values in RL?

-   The Bellman equation calculates the value as the sum of immediate reward plus the discounted value of the state that follows, instead of calculating each value as the sum of the expected return, which is a long process.

##### 3.  Can you explain the difference between a state-value function and an action-value function?

-   A state-value function outputs the expected return if the agent starts at a given state and acts accordingly to the policy forever after. An action-value function outputs the expected return if the agent starts in a given state, takes a given action at that state, and then acts accordingly to the policy forever after.

##### 4.  What are the two types of methods for learning a policy for a value function?

-   The Monte Carlo method and the TD Learning method.

##### 5.  How does the Monte Carlo method differ from the TD Learning method in updating the value function?

-   With the Monte Carlo method, we update the value function from a complete episode, and so we use the actual accurate discounted return of this episode. With the TD Learning method, we update the value function from a step, so we replace Gt that we don’t have with an estimated return called TD target.

##### 6.  How does the concept of value differ from the concept of reward in RL?

-   The value of a state, or a state-action pair is the expected cumulative reward our agent gets if it starts at this state (or state-action pair) and then acts accordingly to its policy. The reward is the feedback we get from the environment after performing an action at a state.

##### 7.  What is Q-Learning and what does it train?

-   Q-Learning is the RL algorithm that trains a Q-function (an action-value function), which internally is a Q-table that contains all the state-action pair values.

##### 8.  Can you explain the relationship between a Q-function, a Q-table, and an optimal policy?

-   The Q-table contains the values of all the state-action pairs, which are calculated by the Q-function. If we have an optimal Q-function, we will have an optimal Q-table and an optimal policy since we know for each state what is the best action to take.

##### 9.  How does Q-Learning determine the best action for a given state?

-   Q-Learning uses the epsilon-greedy strategy, where with a probability of 1-epsilon, the agent selects the action with the highest state-action pair value, and with a probability of epsilon, the agent tries a random action.

##### 10.  How does Q-Learning ensure that the Q-function and Q-table are optimal?

-   Q-Learning updates the Q-table using the Bellman equation, which ensures that the values in the Q-table converge to the optimal values.

##### 11.  Can you explain the process and purpose of the epsilon-greedy strategy in Q-Learning pseudocode?

-   The epsilon-greedy strategy balances exploration and exploitation in Q-Learning. With a probability of 1-epsilon, the agent selects the action with the highest expected reward (exploitation), and with a probability of epsilon, the agent tries a random action (exploration). This allows the agent to explore the environment and learn new information while also making use of the information it has already learned.

##### 12.  How does a neural network work in policy-based methods?

-   In policy-based methods, the policy is usually trained with a neural network to select what action to take given a state. The neural network outputs the action that the agent should take instead of using a value function. Depending on the experience received by the environment, the neural network will be re-adjusted and will provide better actions.

##### 13.  What is the difference between a policy-based method and a value-based method?

-   In policy-based methods, the policy is trained directly to learn which action to take given a state. In value-based methods, rather than learning the policy, we define the policy by hand and we learn a value function. If we have an optimal value function, we will have an optimal policy.

##### 14.  What is the difference between the state-value function and the action-value function in value-based methods?
-   The state-value function calculates for each state the expected return if the agent starts in that state and follows the policy until the end. The action-value function calculates for each state and action pair the expected return if the agent starts in that state and takes an action, then it follows the policy forever after.

##### 15.  How does the epsilon-greedy strategy balance exploration and exploitation?

-   The epsilon-greedy strategy chooses the action with the highest expected reward with a probability of 1-epsilon, and chooses a random action with a probability of epsilon. This allows the agent to explore the environment and learn new information while also making use of the information it has already learned. Epsilon is typically decreased over time to shift focus towards exploitation.

##### 16.  How does the greedy strategy differ from the epsilon-greedy strategy?

-   The greedy strategy always chooses the action that is expected to lead to the highest reward, based on the current knowledge of the environment. It does not include any exploration. The epsilon-greedy strategy balances exploration and exploitation.

##### 17.  In what situations can the greedy strategy be disadvantageous?

-   The greedy strategy can be disadvantageous in environments with uncertainty or unknown optimal actions. It may lead to suboptimal decisions because the agent doesn't explore other options.

##### 18.  How does the epsilon value in epsilon-greedy strategy change over time?

-   Epsilon typically decreases over time as the agent becomes more confident in its knowledge of the environment and the best actions to take. This allows the agent to focus more on exploitation as time goes on.

##### 19.  Can you explain the main idea of the exploration and exploitation trade-off in RL?

-   The exploration and exploitation trade-off in RL refers to the balance between the agent's need to explore the environment and learn new information, and its need to make use of the information it has already learned. The agent must balance these two goals in order to make the best decisions and find the optimal policy.

##### 20.  What is the main idea behind the Policy-based and the Value-based methods?

-   In policy-based methods, the agent learns the policy directly, while in value-based methods, the agent learns a value function, which is then used to determine the best actions to take. The goal in both methods is to find the optimal policy, but they approach it in different ways.

# Unit 3
##### 1.  What is the difference between tabular method and deep Q-Learning?

-   Tabular method is a type of problem in which the state and action spaces are small enough to approximate value functions to be represented as arrays and tables. Q-learning is an example of tabular method since a table is used to represent the value for different state-action pairs. On the other hand, deep Q-Learning is a method that trains a neural network to approximate, given a state, the different Q-values for each possible action at that state. It is used to solve problems when observational space is too big to apply a tabular Q-Learning approach.

##### 2.  How is the temporal limitation addressed in deep Q-Learning?

-   The difficulty presented when the environment state is represented by frames is called temporal limitation. A frame by itself does not provide temporal information. To obtain temporal information, we need to stack a number of frames together.

##### 3.  Can you describe the three phases of deep Q-learning?

-   Phases of Deep Q-Learning are Sampling, Training and Evaluation. In Sampling actions are performed, and observed experience tuples are stored in a replay memory. In Training, batches of tuples are selected randomly and the neural network updates its weights using gradient descent. In Evaluation, the trained model is tested in the environment.

##### 4.  How does experience replay help stabilize deep Q-learning?
-   Experience Replay is a technique used to stabilize deep Q-learning. A replay memory is created to save experiences samples that can be reused during training. This allows the agent to learn from the same experiences multiple times, it makes the agent avoid to forget previous experiences as it get new ones. Random sampling from replay buffer allows to remove correlation in the observation sequences and prevents action values from oscillating or diverging catastrophically.

##### 5.  Can you explain the concept of fixed Q-target in deep Q-learning and how it helps stabilize the training process?

-   In order to calculate the Q-Target we need to estimate the discounted optimal Q-value of the next state by using Bellman equation. The problem is that the same network weigths are used to calculate the Q-Target and the Q-value. This means that everytime we are modifying the Q-value, the Q-Target also moves with it. To avoid this issue, a separate network with fixed parameters is used for estimating the Temporal Difference Target. The target network is updated by copying parameters from our Deep Q-Network after certain C steps.
##### 6. Can you explain the concept of fixed Q-target in deep Q-learning in simple terms? (ChatGPT)
- In deep Q-learning, the goal is to train a neural network to predict the best action to take in a given state, by estimating the Q-value of each action. One of the challenges in this process is that the Q-value estimates can change rapidly, which can make the training process unstable. To address this problem, the fixed Q-target technique is used.
- The idea behind fixed Q-target is to use two separate neural networks during training: one network (the "online" network) is used to select actions and make Q-value predictions, and the other network (the "target" network) is used to calculate the Q-value targets that the online network should aim to match. The target network is "fixed" in the sense that its parameters are not updated during training, they are only copied from the online network at certain intervals.
- By using a fixed target network, the Q-value targets don't change as frequently as the online network's predictions. This makes the training process more stable and less prone to oscillations or divergence.
- --
- *Check out [The Deep Q Algorithm](https://huggingface.co/deep-rl-course/unit3/deep-q-algorithm?fw=pt) by huggingface for more visual explaination for the algorithm.*
- *According to this [Deep Q Learning Tutorial](https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc#:~:text=Deep%20Q%2DLearning%20uses%20Experience,to%20train%20after%20each%20step.) Deep Q-Learning is that the learning process uses 2 neural networks. I encourage you to read it.*
- *My [favourite illustration](https://arshren.medium.com/deep-q-learning-a-deep-reinforcement-learning-algorithm-f1366cf1b53d) to understand that the **target network** is just a copy of the Q-network, which is used to approximate the Q-function.*
-  *An illustration of [DQN architecture](https://towardsdatascience.com/deep-q-network-with-pytorch-146bfa939dfe#:~:text=DQN%20is%20also%20a%20model,estimate%20the%20Q%20value%20function.)*

##### 7. How do we use Double Deep Q-Learning?
When we compute the Q target, we use two networks to decouple the action selection from the target Q value generation. We:
-   Use our _DQN network_ to **select the best action to take for the next state** (the action with the highest Q value).
    
-   Use our _Target network_ to calculate **the target Q value of taking that action at the next state**.
- 
##### 8.  Can you explain the Double DQN method and its role in addressing the problem of Q-value over estimation?

-   Double DQN is a method to handle overstimation of Q-Values. This solution uses two networks to decouple the action selection from the target -Value generation: 
	- DQN Network to select the best action to take for the next state (the action with the highest Q-Value) 
	- Target Network to calculate the target Q-Value of taking that action at the next state. 
- This approach reduces the Q-Values over-estimation, it helps to train faster and have more stable learning. It is done by using the DQN network to select the best action, but using the target network to calculate the Q-value of that action. This prevents the over-estimation of Q-values as the two networks are decoupled, and the DQN network is not used to calculate the Q-values directly.
