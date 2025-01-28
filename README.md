# BeyondTheRainbowICML
Repository for the ICML Paper "BeyondTheRainbow"

In this paper, show Rainbow DQN extended with the following 6 extensions:
<img width="782" alt="rainbow_compare" src="https://github.com/VIPTankz/BeyondTheRainbow2024/assets/41129056/8f6eb599-7662-42cf-8003-49ceeb94ffd9">

Here are our results on the Atari-60 and our ablations on Atari-5:

![BTR60_curve-1](https://github.com/user-attachments/assets/d75bda0d-1e71-4c69-8906-82924377620a)
![DualAblationReducedNoise-1](https://github.com/user-attachments/assets/7fec914f-f8db-4dd5-a135-bdece1ef8b3f)


For our results, we provide a .csv file named results.csv, containing the results for each game. For each game, there are 200 evaluations, one for each million frames. Per evaluation, the given value is the average of 100 episodes.

In order to run our results, first install our environment via the requirements.txt file (all code was tested on Python 3.11.0):

run: pip install -r requirements.txt

Download your correct version of pytorch here: https://pytorch.org/

(We Use PyTorch version 2.1.2, with cuda v1.21)

After installing the environment, you can use main.py to perform runs. The default game is NameThisGame, however you can change this using the command line argument --game "GameName". (ie --game Breakout)

There are also many other command line arguments worth checking out, so have a look at main.py to see.

Also note by default we use 64 parallel environments, and even more for evaluation. This can be quite CPU and RAM intensive, especially on some operating systems such as Windows.
Furthermore, the full Replay Buffer and environments can use up to 45GB of RAM, so beware. If this is an issue, try reducing the number of evaluation environments, training environments or replay buffer size.

Also note that although we use a custom atari environment, this is exactly the same as the standard by default. We also however add a --life_info option, which passes a terminal to the agent on life loss, but does not reset the episode. Using this will drastically improve performance on games with lives.
