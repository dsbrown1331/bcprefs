import numpy as np
import torch
from torch.optim import Adam
import gym
import time

from spinup.utils.test_policy import load_policy_and_env



    #env.plot_entire_trajectory()
if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, help="path to saved model and stuff")
    parser.add_argument('--save_path', type=str, help="path to save demos for later use")
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_rollouts', type=int, default=10, help='how many rollouts eval over')
    parser.add_argument('--random_rollouts', action='store_true', default = False, help = 'generate purely random rollouts')
    args = parser.parse_args()

    #load pretrained policy and env using built in spinning up functionality
    env, get_action = load_policy_and_env(args.load_path)
    demos = []
    for i_episode in range(args.num_rollouts):
        observation = env.reset()
        states = []
        actions = []
        t = 0
        while True:
            env.render()
            # print(observation)
            # action = env.action_space.sample()
            states.append(observation)
            action = get_action(observation)
            if args.random_rollouts:
                #generate completely random actions if flag is set
                action = env.action_space.sample()
            actions.append(action)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            t = t + 1
        states = np.array(states)
        actions = np.array(actions)
        demos.append((states,actions))
    env.close()
    print("generated", len(demos), "demos")

    import pickle

    pickle.dump(demos, open( "demonstrations/{}.p".format(args.save_path), "wb" ) )
