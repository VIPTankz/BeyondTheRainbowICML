# BeyondTheRainbowICML
Repository for the ICML Paper "BeyondTheRainbow"

In this paper, show Rainbow DQN extended with the following 6 extensions:
<img width="782" alt="rainbow_compare" src="https://github.com/VIPTankz/BeyondTheRainbow2024/assets/41129056/8f6eb599-7662-42cf-8003-49ceeb94ffd9">

Here are our results on the Atari-5 and 60 game subsets:

![BTR60_curve-1](https://github.com/user-attachments/assets/d75bda0d-1e71-4c69-8906-82924377620a)
![DualAblation-1](https://github.com/user-attachments/assets/cf4e90fc-df06-46d3-823f-c0102b31c4ba)



For our results, we provide a .csv file named results.csv, containing the results for each game. For each game, there are 200 evaluations, one for each million frames. Per evaluation, there are 100 episode scores.

In order to run our results, first install our environment via the requirements.txt file (all code was tested on Python 3.11.0):

run: pip install -r requirements.txt

Download your correct version of pytorch here: https://pytorch.org/

(We Use PyTorch version 2.1.2, with cuda v1.21)

After installing the environment, you can use main.py to perform runs. The default game is NameThisGame, however you can change this using the command line argument --game "GameName". (ie --game Breakout)

There are also many other command line arguments worth checking out, so have a look at main.py to see.

Also note by default we use 64 parallel environments, and even more for evaluation. This can be quite CPU and RAM intensive.
Furthermore, the full Replay Buffer and environments can use up to 45GB of RAM, so beware. If this is an issue, try reducing the number of evaluation environments, training environments or replay buffer size.

Also note that although we use a custom atari environment, this is exactly the same as the standard by default. We also however add a --life_info option, which passes a terminal to the agent on life loss, but does not reset the episode. Using this will drastically improve performance on games with lives.
