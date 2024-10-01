import numpy as np
import time
from copy import deepcopy
import torch
import gymnasium as gym
import os
import argparse
import multiprocessing as mp
from Agent import Agent, choose_eval_action
from AtariPreprocessingCustom import AtariPreprocessingCustom
from functools import partial
from matplotlib import pyplot as plt

def make_env(envs_create, game, life_info, framestack, repeat_probs):
    return gym.vector.AsyncVectorEnv([lambda: gym.wrappers.FrameStack(
        AtariPreprocessingCustom(gym.make("ALE/" + game + "-v5", frameskip=1, repeat_action_probability=repeat_probs), life_information=life_info), framestack,
        lz4_compress=False) for _ in range(envs_create)], context="spawn")

    #, render_mode="human"


def non_default_args(args, parser):
    result = []
    for arg in vars(args):
        user_val = getattr(args, arg)
        default_val = parser.get_default(arg)
        if user_val != default_val and default_val != "NameThisGame" and arg != "include_evals" and arg != "eval_envs"\
                and arg != "num_eval_episodes" and arg != "analy":

            result.append(f"{arg}={user_val}")
    return ', '.join(result)


def format_arguments(arg_string):
    arg_string = arg_string.replace('=', '')
    arg_string = arg_string.replace('True', '1')
    arg_string = arg_string.replace('False', '0')
    arg_string = arg_string.replace(', ', '_')
    return arg_string


def evaluate_agent(net_state_dict, network_creator, eval_envs, num_eval_episodes, agent_name, testing, game, life_info,
                   n_actions, device, index, framestack, repeat_probs):

    eval_env = make_env(eval_envs, game, life_info, framestack, repeat_probs)
    evals = []
    eval_episodes = 0
    eval_scores = np.array([0 for i in range(eval_envs)])
    eval_observation, eval_info = eval_env.reset()

    eval_net = network_creator()

    # move state dict to gpu - pytorch doesn't allow sharing across threads on gpu
    state_dict_gpu = {k: v.to(device) for k, v in net_state_dict.items()}

    eval_net.load_state_dict(state_dict_gpu)

    # this massively helps speed up training since agents get stuck in some games, causing evals to last a very
    # long time. Also nice to see the difference between 0.00 and 0.01 during evals, like Atari Phoenix.
    if index <= 125:
        rng = 0.01
    else:
        rng = 0.0
    while eval_episodes < num_eval_episodes:

        eval_action = choose_eval_action(eval_observation, eval_net, n_actions, device, rng)
        eval_observation_, eval_reward, eval_done_, eval_trun_, eval_info = eval_env.step(eval_action)
        eval_done_ = np.logical_or(eval_done_, eval_trun_)

        for i in range(eval_envs):
            eval_scores[i] += eval_reward[i]
            if eval_done_[i]:
                eval_episodes += 1
                evals.append(eval_scores[i])
                eval_scores[i] = 0
                if eval_episodes >= num_eval_episodes:
                    break

        eval_observation = eval_observation_

    if not testing:
        fname = agent_name + "Evaluation.npy"
        data = np.load(fname)

        # Update the specified index in the 0th dimension
        data[index] = evals
        print("Evaluation " + str(index + 1) + "M Complete, average score:")
        print(np.mean(evals))

        # Save the updated array back to the file
        np.save(fname, data)


def main():
    parser = argparse.ArgumentParser()

    # environment setup
    parser.add_argument('--game', type=str, default="NameThisGame")
    parser.add_argument('--envs', type=int, default=64)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--rr', type=float, default=1)
    parser.add_argument('--frames', type=int, default=200000000)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--include_evals', type=int, default=1)
    parser.add_argument('--eval_envs', type=int, default=10)
    parser.add_argument('--life_info', type=int, default=0)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--analy', type=int, default=0)
    parser.add_argument('--framestack', type=int, default=4)
    parser.add_argument('--sticky', type=int, default=1)

    # agent setup
    parser.add_argument('--nstep', type=int, default=3)
    parser.add_argument('--vector', type=int, default=1)
    parser.add_argument('--maxpool_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--munch', type=int, default=1)
    parser.add_argument('--munch_alpha', type=float, default=0.9)
    parser.add_argument('--grad_clip', type=int, default=10)

    parser.add_argument('--noisy', type=int, default=1)
    parser.add_argument('--spectral', type=int, default=1)
    parser.add_argument('--iqn', type=int, default=1)
    parser.add_argument('--c51', type=int, default=0)
    parser.add_argument('--maxpool', type=int, default=1)
    parser.add_argument('--arch', type=str, default='impala')
    parser.add_argument('--impala', type=int, default=1)
    parser.add_argument('--discount', type=float, default=0.997)
    parser.add_argument('--per', type=int, default=1)
    parser.add_argument('--taus', type=int, default=8)
    parser.add_argument('--c', type=int, default=500)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--linear_size', type=int, default=512)
    parser.add_argument('--model_size', type=float, default=2)

    parser.add_argument('--double', type=int, default=0)
    parser.add_argument('--ncos', type=int, default=64)
    parser.add_argument('--per_alpha', type=float, default=0.2)
    parser.add_argument('--per_beta_anneal', type=int, default=0)
    parser.add_argument('--layer_norm', type=int, default=0)
    parser.add_argument('--eps_steps', type=int, default=2000000)
    parser.add_argument('--eps_disable', type=int, default=1)
    parser.add_argument('--activation', type=str, default="relu")

    args = parser.parse_args()

    arg_string = non_default_args(args, parser)
    formatted_string = format_arguments(arg_string)
    print(formatted_string)

    game = args.game
    envs = args.envs
    bs = args.bs
    rr = args.rr
    c = args.c
    lr = args.lr
    life_info = args.life_info
    num_eval_episodes = args.num_eval_episodes
    analy = args.analy
    framestack = args.framestack
    sticky = args.sticky
    repeat_probs = 0 if not sticky else 0.25

    nstep = args.nstep
    maxpool_size = args.maxpool_size
    noisy = args.noisy
    spectral = args.spectral
    munch = args.munch
    munch_alpha = args.munch_alpha
    grad_clip = args.grad_clip
    arch = args.arch
    iqn = args.iqn
    double = args.double
    dueling = args.dueling
    impala = args.impala
    discount = args.discount
    linear_size = args.linear_size
    per = args.per
    taus = args.taus
    model_size = args.model_size
    frames = args.frames // 4
    ncos = args.ncos
    maxpool = args.maxpool
    vector = args.vector
    per_alpha = args.per_alpha
    per_beta_anneal = args.per_beta_anneal
    layer_norm = args.layer_norm
    c51 = args.c51
    eps_steps = args.eps_steps
    eps_disable = args.eps_disable
    activation = args.activation

    if not vector:
        lr = 5e-5
        envs = 1
        bs = 16
        rr = 0.25

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
        counter = 0
        while True:
            if counter == 0:
                new_dir_name = agent_name
            else:
                new_dir_name = f"{agent_name}_{counter}"
            if not os.path.exists(new_dir_name):
                break
            counter += 1
        os.mkdir(new_dir_name)
        print(f"Created directory: {new_dir_name}")
        os.chdir(new_dir_name)

    # create blank evaluation file
    fname = agent_name + "Evaluation.npy"
    if not testing:
        np.save(fname, np.zeros((args.frames // 1000000, num_eval_episodes)))

    if testing:
        num_envs = 4
        eval_envs = 2
        eval_every = 8000
        num_eval_episodes = 4
        n_steps = 8000
        bs = 32
    else:
        num_envs = envs
        eval_envs = args.eval_envs
        n_steps = frames
        eval_every = 250000
    next_eval = eval_every

    print("Currently Playing Game: " + str(game))

    gpu = "0"
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    env = make_env(num_envs, game, life_info, framestack, repeat_probs)
    print(env.observation_space)
    print(env.action_space[0])
    n_actions = env.action_space[0].n

    agent = Agent(n_actions=env.action_space[0].n, input_dims=[framestack, 84, 84], device=device, num_envs=num_envs,
                  agent_name=agent_name, total_frames=n_steps, testing=testing, batch_size=bs, rr=rr, lr=lr,
                  maxpool_size=maxpool_size, target_replace=c,
                  noisy=noisy, spectral=spectral, munch=munch, iqn=iqn, double=double, dueling=dueling, impala=impala,
                  discount=discount, per=per, taus=taus,
                  model_size=model_size, linear_size=linear_size, ncos=ncos, maxpool=maxpool, replay_period=num_envs,
                  analytics=analy, framestack=framestack, arch=arch, per_alpha=per_alpha,
                  per_beta_anneal=per_beta_anneal, layer_norm=layer_norm, c51=c51, eps_steps=eps_steps,
                  eps_disable=eps_disable, activation=activation, n=nstep, munch_alpha=munch_alpha, grad_clip=grad_clip)

    scores_temp = []
    steps = 0
    last_steps = 0
    last_time = time.time()
    episodes = 0
    current_eval = 0
    scores_count = [0 for i in range(num_envs)]
    scores = []
    observation, info = env.reset()
    processes = []

    if testing:
        from torchsummary import summary
        summary(agent.net, (framestack, 84, 84))

    while steps < n_steps:
        steps += num_envs
        action = agent.choose_action(observation)
        env.step_async(action)
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
            agent.store_transition(observation[stream], action[stream], reward[stream], observation_[stream],
                                   terminal_in_buffer, stream=stream)

        observation = observation_

        if steps % 1200 == 0 and len(scores) > 0:
            avg_score = np.mean(scores_temp[-50:])
            if episodes % 1 == 0:
                print('{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f} games {}'
                      .format(agent_name, game, avg_score, steps, (steps - last_steps) / (time.time() - last_time), episodes),
                      flush=True)
                last_steps = steps
                last_time = time.time()

        # Evaluation
        if steps >= next_eval or steps >= n_steps:
            print("Evaluating")

            # Save model
            if not testing and (current_eval + 1) % 10 == 0:
                agent.save_model()

            fname = agent_name + "Experiment.npy"
            if not testing:
                np.save(fname, np.array(scores))

            if include_evals:

                # wait for our evaluations to finish before we start the next evaluation

                for process in processes:
                    process.join()

                agent.disable_noise(agent.net)
                net_state_dict = deepcopy({k: v.cpu() for k, v in agent.net.state_dict().items()})
                network_creator = deepcopy(agent.network_creator_fn)

                # Start evaluation in a separate process
                eval_process = mp.Process(target=evaluate_agent,
                                          args=(net_state_dict, network_creator, eval_envs, num_eval_episodes, agent_name, testing, game,
                                                life_info, n_actions, device, current_eval, framestack, repeat_probs))
                eval_process.start()
                processes.append(eval_process)

                current_eval += 1

            next_eval += eval_every

    # wait for our evaluations to finish before we quit the program
    for process in processes:
        process.join()

    print("Evaluations finished, job completed successfully!")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
