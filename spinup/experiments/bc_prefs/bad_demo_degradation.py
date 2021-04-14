##Tests how performance degrades as more and more good demos are replaced with bad demos

import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import gym
import os
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from spinup.algos.pytorch.bc.bc import train_bc_policy, evaluate_policy

torchify = lambda x: torch.FloatTensor(x).to(torch.device('cpu')) 
 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help='environment for bc')
    parser.add_argument('--good_load_path', type=str, help='path to recorded good demonstrations')
    parser.add_argument('--bad_load_path', type=str, help='path to recorded bad demonstrations')
    parser.add_argument('--seed', '-s', type=int, default=4)
    parser.add_argument('--pi_lr', type=float, default=1e-2, help="learning rate for policy")
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--clone', action="store_true", help="do behavior cloning")
    # parser.add_argument('--num_demos', type=int, default=6)
    parser.add_argument('--BC_iters', type=int, default=1000)
    parser.add_argument('--save_path', type=str, help='path to record results')


    parser.add_argument('--num_rollouts', type=int, default=100)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env)
    env.seed(args.seed)
    #get demos
    # demo_obs, demo_acs = env.get_demos(args.num_demos)
    # Load the dictionary back from the pickle file.
    import pickle

    good_demos = pickle.load( open( args.good_load_path, "rb" ) )
    bad_demos = pickle.load( open( args.bad_load_path, "rb" ) )


    #grab everything before .p and after the last /
    good_name = args.good_load_path.split("/")[-1].split(".")[0]
    bad_name = args.bad_load_path.split("/")[-1].split(".")[0]
   
    save_file_name = "{}vs{}".format(good_name, bad_name)

    #for now we'll assume the same number of demos (good vs bad)
    assert len(good_demos)  == len(bad_demos)

    print("loaded good demos", len(good_demos))
    print("loaded bad demos", len(bad_demos))

    results_bad_degredation = np.zeros((args.num_rollouts, len(good_demos)+1))
    #start with all good demos and slowly add bad demos
    for num_bad_demos in range(0,len(bad_demos)+1):

        num_good_demos = len(good_demos) - num_bad_demos 
        training_demos = []
        training_demos.extend(bad_demos[:num_bad_demos])
        training_demos.extend(good_demos[:num_good_demos])
        print("good {} bad {}".format(num_good_demos, num_bad_demos))
        print(len(training_demos))


        #collect all the demos into two long arrays
        demo_obs = []
        demo_acs = []
        for states,actions in training_demos:
            for s in states:
                demo_obs.append(s)
            for a in actions:
                demo_acs.append(a)
        print("states", len(demo_obs), "actions", len(demo_acs))

        #train policy 
        pi = train_bc_policy(demo_obs, demo_acs, env, args.BC_iters, args.hid_size, args.num_layers, args.pi_lr)

        # Visualize policy
        ep_returns = evaluate_policy(env, pi, args.num_rollouts, viz=False)
        print(np.mean(ep_returns))
        results_bad_degredation[:,num_bad_demos] = ep_returns

    #save results
    import pickle
    import os.path

    print(save_file_name)
    print(os.path.join(args.save_path, save_file_name))
    pickle.dump(results_bad_degredation, open( os.path.join(args.save_path, "{}.p".format(save_file_name)), "wb" ))


    # x = range(0,11)
    # ave_perf = np.mean(results_bad_degredation, axis=0)
    # std_perf = np.std(results_bad_degredation, axis=0)
    # plt.xlabel('num bad demos')
    # plt.ylabel('ave performance of bc policy')
    # plt.fill_between(x, ave_perf - std_perf , ave_perf + std_perf, facecolor='green', alpha=0.5)
    # plt.plot(x, ave_perf, 'g')
    # plt.show()    
