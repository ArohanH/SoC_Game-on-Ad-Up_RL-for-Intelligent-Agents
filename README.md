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
      
  ## Mentors
    - Siddhartha Rajeev (7619527676)
    - Arohan Hazarika (9678969740)

    Feel free to contact us!
