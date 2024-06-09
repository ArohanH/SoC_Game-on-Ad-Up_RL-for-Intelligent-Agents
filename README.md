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
  ## Week2 
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
        Create a 4×4 grid world environment.
        Place reward pots at specified positions.
        Define rewards for reaching each pot.
        Define transition dynamics (deterministic)
      
      
  ###### Task 2: Value Function Computation:
      Initialize the value function for the states.
      Implement the Bellman equation for value functions.
      Perform policy iteration (evaluation + improvement) to update the value function and policy until convergence.
      
  ###### Task 3: Optimal Policy:
      Display the optimal policy for navigating the grid as computed through policy iteration.
      
  ###### Task 4: Testing and Visualization:
      
      Test the implementation by running the agent in the grid world.
      Visualize the optimal policy and the path taken by the agent to collect rewards.
      
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
  ## Mentors
    - Siddhartha Rajeev (7619527676)
    - Arohan Hazarika (9678969740)

    Feel free to contact us!
