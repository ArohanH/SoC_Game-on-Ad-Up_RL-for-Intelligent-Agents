# SoC-Game On, Ad Up: RL for Intelligent Agents
This is the official repository for the Seasons of Code Project "Game On, Ad Up: RL for Intelligent Agents" under WnCC, IIT Bombay
  ## Week 1
  ### Assignment 
  - The problem is to look at a multi armed bandit. This bandit has 6 arms(i.e., at each time-step you are offered to choose between 6 options
    as explained below. The agent doesn’t know about the actual probability distributions, the agent can only see 6 buttons and has to figure out an optimal strategy
  - For exploration vs exploitation, you are expected to use ε-Greedy Algorithm
  - One episode is supposed to be 100 time-steps long. Take values of ε as 0.1, 0.01 and 0. Plot graphs of “Reward at the end of episode” vs “Episode” 
    for each of these three epsilon greedy algorithms. Simulate at least 1K episodes. If possible, plot the 3 curves in the same graph. We expect you to play around with         the values of different parameters
  #### The 6 options
    - Gaussian Distribution with mean reward 0 and variance 1
    - Fair Coin Toss with a reward of +3 for head and -4 for tail
    - Poisson Distribution with variance in reward 2
    - Gaussian Distribution with mean reward +1 and variance 2
    - Exponenetial Distribution with mean reward +1
    - The last but not the least, this button chooses any of the previous options with equal probability
  #### Few Instructions
  - This task is expected to be done in Python. There are a few libraries you must be familiar with before you start coding. While the implementation aspect will be rather
    simple, this week’s primary target is to get you familiarized with NumPy and Matplotlib 
  - You can use numpy functions for generating the different probability distributions
  - Submission Deadline will be 8th of June
      
  ## Mentors
    - Siddhartha Rajeev
    - Arohan Hazarika
