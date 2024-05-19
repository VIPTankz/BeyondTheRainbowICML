# BeyondTheRainbow2024
Repository for the 2024 Paper "BeyondTheRainbow"

In this paper, show Rainbow DQN extended with the following 6 extensions:
<img width="782" alt="rainbow_compare" src="https://github.com/VIPTankz/BeyondTheRainbow2024/assets/41129056/8f6eb599-7662-42cf-8003-49ceeb94ffd9">

Here are our results on the Atari-5 and 15 game subsets:

results go here

For our results, we provide a .csv file named results.csv, containing the results for each game. For each game, there are 200 evaluations, one for each million frames. Per evaluation, there are 100 episode scores.

In order to run our results, first install our environment via the requirements.txt file (all code was tested on Python 3.11.0):
(Also note PyTorch has different versions for different platforms, so you may need to install it from there website: https://pytorch.org/. We Use PyTorch version 2.1.2, with cuda v1.21)

After installing the environment, you can use main.py to perform runs. The default game is BattleZone, however you can change this using the command line argument --game "GameName".

There are also many other command line arguments worth checking out, so have a look at main.py to see.

Also note by default we use 64 parallel environments, and even more for evaluation. This can be quite CPU intensive.
Furthermore, the full Replay Buffer and environments can use up to 45GB of RAM, so beware. If this is an issue, try reducing the number of evaluation environments, training environments or replay buffer size.
