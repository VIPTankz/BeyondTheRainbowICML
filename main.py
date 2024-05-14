
import numpy as np
import time
from copy import deepcopy
import sys
import torch
import gymnasium as gym
import wandb
import os
import argparse, sys
from Agent import Agent
from AtariPreprocessingCustom import AtariPreprocessingCustom
from torch.profiler import profile, record_function, ProfilerActivity
def make_env(envs_create):

    return gym.vector.AsyncVectorEnv([lambda: gym.wrappers.FrameStack(
        AtariPreprocessingCustom(gym.make("ALE/" + game + "-v5", frameskip=1), life_information=life_info), 4,
        lz4_compress=False) for _ in range(envs_create)], context="spawn")

def non_default_args(args, parser):
    result = []
    for arg in vars(args):
        user_val = getattr(args, arg)
        default_val = parser.get_default(arg)

        # Check if the user provided value differs from the default
        if user_val != default_val and default_val != "BattleZone" and arg != "include_evals" and arg != "eval_envs" and arg != "num_eval_episodes":
            result.append(f"{arg}={user_val}")
    return ', '.join(result)


def format_arguments(arg_string):
    # Remove all '=' signs
    arg_string = arg_string.replace('=', '')

    # Replace "True" with "1" and "False" with "0"
    arg_string = arg_string.replace('True', '1')
    arg_string = arg_string.replace('False', '0')
    arg_string = arg_string.replace(', ', '_')

    return arg_string

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default="BattleZone")
    parser.add_argument('--envs', type=int, default=64)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--rr', type=float, default=1)
    parser.add_argument('--frames', type=int, default=200000000)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--include_evals', type=int, default=0)
    parser.add_argument('--eval_envs', type=int, default=16)
    parser.add_argument('--life_info', type=int, default=0)
    parser.add_argument('--num_eval_episodes', type=int, default=100)

    parser.add_argument('--vector', type=int, default=1)  # This overwrites many args down below

    parser.add_argument('--maxpool_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--ema_tau', type=float, default=2.5e-4)
    parser.add_argument('--munch', type=int, default=1)

    # the way parser.add_argument handles bools in dumb so we use int 0 or 1 instead
    parser.add_argument('--noisy', type=int, default=1)
    parser.add_argument('--spectral', type=int, default=1)
    parser.add_argument('--spectral_lin', type=int, default=0)
    parser.add_argument('--iqn', type=int, default=1)
    parser.add_argument('--maxpool', type=int, default=1)

    parser.add_argument('--impala', type=int, default=1)
    parser.add_argument('--discount', type=float, default=0.997)

    parser.add_argument('--per', type=int, default=1)
    parser.add_argument('--taus', type=int, default=8)
    parser.add_argument('--c', type=int, default=500)  # this is the target replace
    parser.add_argument('--dueling', type=int, default=1)

    # features still in testing

    parser.add_argument('--linear_size', type=int, default=512)
    parser.add_argument('--model_size', type=int, default=2)
    parser.add_argument('--tr', type=int, default=0)

    # not applicable when using munchausen
    parser.add_argument('--double', type=int, default=0)

    # likely dead improvements
    parser.add_argument('--adamw', type=int, default=0)
    parser.add_argument('--lr_decay', type=int, default=0)
    parser.add_argument('--discount_anneal', type=int, default=0)
    parser.add_argument('--ema', type=int, default=0)
    parser.add_argument('--ncos', type=int, default=64)

    args = parser.parse_args()

    arg_string = non_default_args(args, parser)
    formatted_string = format_arguments(arg_string)
    print(formatted_string)

    game = args.game
    envs = args.envs
    bs = args.bs
    rr = args.rr
    ema = args.ema
    tr = args.tr
    c = args.c
    ema_tau = args.ema_tau
    lr = args.lr
    life_info = args.life_info
    num_eval_episodes = args.num_eval_episodes

    maxpool_size = args.maxpool_size

    noisy = args.noisy
    spectral = args.spectral
    spectral_lin = args.spectral_lin
    munch = args.munch
    iqn = args.iqn
    double = args.double

    dueling = args.dueling
    impala = args.impala
    discount = args.discount

    linear_size = args.linear_size

    adamw = args.adamw
    lr_decay = args.lr_decay

    per = args.per
    taus = args.taus
    model_size = args.model_size

    frames = args.frames // 4  # "frames" is the actual number of frames. This variable frames represents steps
    # apologies for confusing name

    ncos = args.ncos

    discount_anneal = args.discount_anneal
    maxpool = args.maxpool

    # atari-3 : Battle Zone, Name This Game, Phoenix
    # atari-5 : Battle Zone, Double Dunk, Name This Game, Phoenix, Qbert

    vector = args.vector
    if not vector:
        lr = 5e-5
        envs = 4
        bs = 16
        rr = 1

    lr_str = "{:e}".format(lr)
    lr_str = str(lr_str).replace(".", "").replace("0", "")
    frame_name = str(int(args.frames / 1000000)) + "M"

    include_evals = bool(args.include_evals)

    agent_name = "BTR_" + game + frame_name

    if len(formatted_string) > 2:
        agent_name += '_' + formatted_string

    print("Agent Name:" + str(agent_name))
    testing = args.testing

    if not testing:
        ###################### Making Dir Code
        # Initialize a counter to keep track of the suffix
        counter = 0

        # Loop until you find a directory name that doesn't exist
        while True:
            # Construct the directory name with the current counter
            if counter == 0:
                new_dir_name = agent_name
            else:
                new_dir_name = f"{agent_name}_{counter}"

            # Check if the directory already exists
            if not os.path.exists(new_dir_name):
                break

            # If it exists, increment the counter and try again
            counter += 1

        os.mkdir(new_dir_name)
        print(f"Created directory: {new_dir_name}")
        os.chdir(new_dir_name)

        #############################

    # atari-3 : Battle Zone, Name This Game, Phoenix
    # atari-5 : Battle Zone, Double Dunk, Name This Game, Phoenix, Q*Bert

    if testing:
        num_envs = 4
        eval_envs = 2
        eval_every = 10000
        num_eval_episodes = 5
        n_steps = 25000
        bs = 16
    else:
        num_envs = envs
        eval_envs = args.eval_envs
        n_steps = frames
        eval_every = 250000  # evaluate every 250k steps, equal to 1M Frames

    next_eval = eval_every

    print("Currently Playing Game: " + str(game))

    gpu = "0"
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    env = make_env(num_envs)

    print(env.observation_space)
    print(env.action_space[0])

    agent = Agent(n_actions=env.action_space[0].n, input_dims=[4, 84, 84], device=device, num_envs=num_envs,
                  agent_name=agent_name, total_frames=n_steps, testing=testing, batch_size=bs, rr=rr, lr=lr,
                  maxpool_size=maxpool_size, ema=ema, trust_regions=tr, target_replace=c, ema_tau=ema_tau,
                  noisy=noisy, spectral=spectral, munch=munch, iqn=iqn, double=double, dueling=dueling, impala=impala,
                  discount=discount, adamw=adamw, discount_anneal=discount_anneal, lr_decay=lr_decay,
                  per=per, taus=taus, model_size=model_size, linear_size=linear_size,
                  spectral_lin=spectral_lin, ncos=ncos, maxpool=maxpool)


    scores_temp = []
    steps = 0
    last_steps = 0
    last_time = time.time()
    episodes = 0
    start = time.time()

    evals_total = []

    scores_count = [0 for i in range(num_envs)]

    dormants = []
    param_norms = []

    scores = []
    done = False
    observation, info = env.reset()

    while steps < n_steps:
        steps += num_envs

        action = agent.choose_action(observation)  # this takes and return batches

        env.step_async(action)

        # this is placed here so learning takes place while step is happening
        agent.learn()

        observation_, reward, done_, trun_, info = env.step_wait()

        done_ = np.logical_or(done_, trun_)

        for i in range(num_envs):
            scores_count[i] += reward[i]

            if done_[i]:
                episodes += 1
                scores.append([scores_count[i], steps])
                scores_temp.append(scores_count[i])

                scores_count[i] = 0

        reward = np.clip(reward, -1., 1.)

        for stream in range(num_envs):
            terminal_in_buffer = done_[stream] or info["lost_life"][stream]

            agent.store_transition(observation[stream], action[stream], reward[stream], terminal_in_buffer, stream=stream)
            # before terminal_in_buffer was just done_[stream].

        observation = observation_

        for stream in range(num_envs):
            if done_[stream]:
                observation[stream] = info["final_observation"][stream]

        if steps % 1200 == 0 and len(scores) > 0:

            avg_score = np.mean(scores_temp[-50:])

            if episodes % 1 == 0:
                print('{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f}'
                      .format(agent_name, game, avg_score, steps, (steps - last_steps) / (time.time() - last_time)), flush=True)
                last_steps = steps
                last_time = time.time()

        # Evaluation
        if steps >= next_eval or steps >= n_steps:

            print("Evaluating")

            # save model
            if not testing:
                agent.save_model()

            fname = agent_name + "Experiment.npy"
            if not testing:
                np.save(fname, np.array(scores))

            if include_evals:
                if 'eval_env' not in globals():
                    eval_env = make_env(eval_envs)

                agent.set_eval_mode()
                evals = []
                eval_episodes = 0
                eval_scores = np.array([0 for i in range(eval_envs)])
                eval_observation, eval_info = eval_env.reset()

                evals_started = [i for i in range(eval_envs)]

                while eval_episodes < num_eval_episodes:

                    eval_action = agent.choose_action(eval_observation)  # this takes and return batches

                    eval_observation_, eval_reward, eval_done_, eval_trun_, eval_info = eval_env.step(eval_action)

                    # TRUNCATION NOT IMPLEMENTED
                    eval_done_ = np.logical_or(eval_done_, eval_trun_)

                    for i in range(eval_envs):
                        eval_scores[i] += eval_reward[i]

                        if eval_done_[i]:
                            if evals_started[i] < num_eval_episodes:
                                evals_started[i] = max(evals_started) + 1
                                eval_episodes += 1

                                evals.append(eval_scores[i])

                                eval_scores[i] = 0

                        if len(evals) == num_eval_episodes:
                            break

                    eval_observation = eval_observation_

                    for stream in range(eval_envs):
                        if eval_done_[stream]:
                            eval_observation[stream] = eval_info["final_observation"][stream]

                evals_total.append(evals)
                fname = agent_name + "Evaluation.npy"
                if not testing:
                    np.save(fname, np.array(evals_total))

                agent.set_train_mode()

            next_eval += eval_every
