# rl_trading
Trading Equities with RL

Howdy gang, thanks for checking out the repository

### Files and Folders
- Data: stored_data
  - data files are the 1min tickers for a variety of equities
- Method for getting the data: data_getter.py and data_getter.ipynb
  - data_getter.py 
    - Has some functions called so it can just be run
    - I have mine automatically running every day to strip the 1 min tickers after close. This was tricky chrontab is hard. lemme know if you wanna set this up
  - data_getter.ipynb
    - ipython file with same functionality as the python file
    - For your intial pulling down of data, you can pull down a weeks worth of 1min ticker data
    - additionally you can change which equities you want in the basket
- Functionality
  - network_builds.py
    - network classes -- actor and critics
  - data_builder.py 
    - class for building dictionaries of data for different equities if you decide to run a multiple actor approach
  - environments.py
    - the environment class where your actor steps throught the "environment" i.e. updates your money, moves to next ticker etc.
  - supp_funcs.py 
    - supplementary functions for the training process
  - model_trainers.py
    - training classes where the actual model training happens
    - requires the functions from supp_functions they update the actor and critic policies
  - plotter.py
    - just plotting functionality to see results
  - run_ppo.ipynb
    - ipython file for putting everything together 
### Links
##### read the first two links, basically a must. The third was the starting point for my code and has other RL implementations 
- [Great Blog](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html) with an intro to RL, some of the types, and helpful links
- [PPO](https://arxiv.org/pdf/1707.06347.pdf) paper, important for understanding exactly how ppo works 
- Excellent [github](https://github.com/higgsfield/RL-Adventure-2) for some rl code that is written pretty simplistically and in pytorch. All the code is for the OpenAI opengym environment, but the basics are super helpful for understanding the algorithms
- This was a method used by the original implementer of the PPO algorithm, it is generally accepted to improve performance for RL algos [GEA](https://arxiv.org/pdf/1506.02438.pdf)
