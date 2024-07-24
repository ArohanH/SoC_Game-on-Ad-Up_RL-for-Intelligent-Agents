# SoC-Game On, Ad Up: RL for Intelligent Agents
This is the official repository for the Seasons of Code Project "Game On, Ad Up: RL for Intelligent Agents" under WnCC, IIT Bombay
  ## General Resources:
  - David Silver's Google DeepMind RL Playlist: https://www.youtube.com/playlist?list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-
  - Sutton and Barto's legendary RL textbook: https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf
  - OpenAI's 'Spinning Up' RL Explanations: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html

  ## Week 1
  ### Assignment 
  - The problem is to look at a multi armed bandit. This bandit has 6 arms(i.e., at each time-step you are offered to choose between 6 options
    as explained below. The agent doesn’t know about the actual probability distributions, the agent can only see 6 buttons and has to figure out an optimal strategy
  - For exploration vs exploitation, you are expected to use ε-Greedy Algorithm
  - One episode is supposed to be 100 time-steps long. Take values of ε as 0.1, 0.01 and 0. Plot graphs of “Reward at the end of episode” vs “Episode” 
    for each of these three epsilon greedy algorithms. Simulate at least 1K episodes. If possible, plot the 3 curves in the same graph. We expect you to play around with         the values of different parameters.

  Note that this is covered in chapter 2 of the textbook.
  #### The 6 options
    - Gaussian Distribution with mean reward 0 and variance 1
    - Fair Coin Toss with a reward of +3 for head and -4 for tail
    - Poisson Distribution with variance in reward 2
    - Gaussian Distribution with mean reward +1 and variance 2
    - Exponential Distribution with mean reward +1
    - The last but not the least, this button chooses any of the previous options with equal probability
  #### Few Instructions
  - This task is expected to be done in Python. There are a few libraries you must be familiar with before you start coding. While the implementation aspect will be rather
    simple, this week’s primary target is to get you familiarized with NumPy and Matplotlib 
  - Feel free to use the numpy random module documentation at https://numpy.org/doc/1.16/reference/routines.random.html to generate random samples
  - Furthermore, you can find the documentation for plotting functions at https://matplotlib.org/stable/api/index
  - The submission deadline will be 23:59, 8th June 2024
  ## Week 2 
  ### Assignment
  Reinforcement Learning with Bellman Equations in a Grid World

  #### Objective:
  To understand and implement concepts such as the Bellman equations, value functions, and action-value functions in a grid-based environment. Use these concepts to solve a 4×4 grid world problem where the objective is to maximize cumulative rewards collected from various reward pots.
      
  #### Background:
  In this assignment, you'll be working on a 4×4 grid world environment. The grid world contains reward pots placed at different locations, each offering a specific reward. The agent's goal is to navigate the grid world, starting from a given position and maximize the total reward by collecting from these pots. The agent can move in four directions: up, down, left, and right, unless blocked by the grid boundary walls.
      
  #### Concepts Covered:
        - Bellman Equations: Fundamental equations in dynamic programming for finding the optimal policy.
        - Value Functions: Measures the expected cumulative reward from each state.
        - Action-Value Functions: Measures the expected cumulative reward from each state-action pair.
      
  #### Assignment Details:
      
  ##### Grid World Setup:
      The grid is a 4×4 matrix.
      Each cell in the grid can contain a reward pot, be empty, or be a starting point.
      The reward values are provided in a separate configuration file or as a 2D array.
      
      
  ##### Objective:
      
  Implement policy improvement and policy evaluation to determine the optimal policy for navigating the grid world. Use Bellman equations to update the value function.
      
  ##### Tasks:
      
  ###### Task 1: Environment Setup:
  - Create a 4×4 grid world environment.
  - Place reward pots at specified positions.
  - Define rewards for reaching each pot.
  - Define transition dynamics (deterministic)
      
      
  ###### Task 2: Value Function Computation:
  - Initialize the value function for the states.
  - Implement the Bellman equation for value functions.
  - Perform policy iteration (evaluation + improvement) to update the value function and policy until convergence.
      
  ###### Task 3: Optimal Policy:
  Display the optimal policy for navigating the grid as computed through policy iteration.
      
  ###### Task 4: Testing and Visualization:
      
  - Test the implementation by running the agent in the grid world.
  - Visualize the optimal policy and the path taken by the agent to collect rewards.
      
  ##### Constraints:
      
  The agent cannot move outside the grid. Transitions are deterministic
       
  ##### Things we're looking for:
      
        - Correctness: Proper implementation of the Bellman equations, value functions in policy (improvement+evaluation)
        - Efficiency: Optimized code that converges reasonably quickly.
        - Documentation: Clear explanation of the code and methodology through comments
        - Visualization: Accurate and informative visualization of the grid world and policy.
        - Creativity: Any additional features or improvements (optional - just have fun with it)
      
  ##### Sample Grid Configuration:
      
      [
        [0,  0,  0,  10],
        [0, -1,  0,   0],
        [0,  0,  0,   0],
        [1,  0,  0,   5]
      ]
      
        - 10 and 5 are reward pots.
        - -1 is a penalty or obstacle.
        - 0 represents empty cells.

  By the end of this assignment, you will have a solid understanding of how to use the Bellman equations to solve reinforcement learning problems in grid-based    environments!
  The submission deadline will be 23:59, 16th June 2024
  ## Week 3
  ### Assignment
  - This week's assignment is based on RL algorithms like Monte-Carlo and Q Learning, you have to apply them on Open AI Gym's "Taxi" Environment.
  - The goal is to move the taxi to the passenger’s location, pick up the passenger, move to the passenger’s desired destination, and drop off the passenger. Once the     passenger is dropped off, the episode ends.
  - This game is simple because both the action and observation spaces are discrete. 
  - Build a Monte Carlo algorithm and a Q-Learning algorithm to attempt to solve this game. Play around with the parameters and try to find the parameters which help you   come up with faster converging algorithms. Make a plot which shows the Monte Carlo cumulative reward and the Q Learning cumulative reward for each episode, for 3000 episodes.
  - Please follow the documentation for information related to rewards, state and action spaces: https://gymnasium.farama.org/environments/toy_text/taxi/
## Week 4
  ### Assignment
  #### Problem Statement: MNIST Digit Recognition Using Convolutional Neural Networks in TensorFlow
  
  #### Overview:
  The MNIST (Modified National Institute of Standards and Technology) dataset is a widely-used benchmark for evaluating machine learning algorithms, particularly in the field of image classification. It consists of 70,000 grayscale images of handwritten digits from 0 to 9, with each image being 28x28 pixels in size. The goal is to correctly identify the digit represented in each image.
  
  #### Objective:
  Develop a robust and efficient convolutional neural network (CNN) using TensorFlow to accurately classify the handwritten digits in the MNIST dataset. The model should be able to generalize well to unseen data, achieving high accuracy on both training and test sets.
  
  #### Requirements:
  ###### 1. Data Preprocessing: 
  - Normalize the pixel values to a range of 0 to 1.
  - Convert the labels to one-hot encoded vectors.
  
  ###### 2. Model Architecture:
  - Design a CNN architecture that effectively captures spatial hierarchies in the images.
  - Experiment with different numbers of convolutional layers, filter sizes, activation functions, and pooling layers to find the optimal architecture.
  
  ###### 3. Training Process:
  - Implement the model training using TensorFlow.
  - Choose appropriate optimization algorithms (e.g., Adam) and loss functions (e.g., categorical cross-entropy).
  
  ###### 4. Evaluation Metrics:
  - Assess the model's performance using accuracy as the primary metric.
  - Additionally, calculate confusion matrices and classification reports to evaluate the performance across different digit classes.
  
  ###### 5. Hyperparameter Tuning:
  - Perform hyperparameter tuning to optimize the model's parameters, such as learning rate, batch size, and number of epochs.
  
  ###### 6. Visualization:
  - Provide visualizations of training and validation accuracy/loss over epochs.

  You can learn about neural networks and convolutions on any YouTube channel of your choosing because it is a very ubiquitous subject available easily!
  ## Week 4
  ### Assignment
  #### Problem Statement: Playing Atari Games with Deep Reinforcement Learning
  
  #### Overview:
  This is the last assignment so there won't be too many hints! Pick your favourite Atari game from [here](https://www.gymlibrary.dev/environments/atari/index.html) and using the example of Andrej Karpathy's [blog](https://karpathy.github.io/2016/05/31/rl/) create a neural network to play the Atari game from pixels!
  
  #### Few Instructions:
  - Kindly go through Lectures 4 and 5 of David Silver's playlist to cover the theory part for Reinforcement Learning Algorithms 
  - Submission Deadline will be 23:59, 30th June, 2024
  ## Mentors
    - Siddhartha Rajeev (7619527676)
    - Arohan Hazarika (9678969740)

    Feel free to contact us!
