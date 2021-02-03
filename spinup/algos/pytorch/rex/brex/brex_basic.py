import argparse

# Use hand tuned feature count vectors for testing the MCMC convergence and distributions.
# Doesn't actually use any kind of real domain.


import numpy as np
import random
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import pickle
#from spinup.algos.pytorch.brex.model_evaluator import model_eval




def calc_linearized_pairwise_ranking_loss(last_layer, pairwise_prefs, demo_cnts, confidence):
    '''use (i,j) indices and precomputed feature counts to do faster pairwise ranking loss'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    #print(device)
    #don't need any gradients
    with torch.no_grad():

        #do matrix multiply with last layer of network and the demo_cnts
        #print(list(reward_net.fc2.parameters()))

        #print(linear)
        #print(bias)
        weights = torch.from_numpy(last_layer) #append bias and weights from last fc layer together
        demo_cnts_tensor = torch.from_numpy(demo_cnts)
        #print('weights',weights)
        #print('demo_cnts', demo_cnts)
        demo_returns = confidence * torch.mv(demo_cnts_tensor, weights)


        loss_criterion = nn.CrossEntropyLoss(reduction='sum') #sum up losses
        cum_log_likelihood = 0.0
        outputs = torch.zeros(len(pairwise_prefs),2) #each row is a new pair of returns
        for p, ppref in enumerate(pairwise_prefs):
            i,j = ppref
            outputs[p,:] = torch.tensor([demo_returns[i], demo_returns[j]])
        labels = torch.ones(len(pairwise_prefs)).long()

        #outputs = outputs.unsqueeze(0)
        #print(outputs)
        #print(labels)
        cum_log_likelihood = -loss_criterion(outputs, labels)
            #if labels == 0:
            #    log_likelihood = torch.log(return_i/(return_i + return_j))
            #else:
            #    log_likelihood = torch.log(return_j/(return_i + return_j))
            #print("ll",log_likelihood)
            #cum_log_likelihood += log_likelihood
    return cum_log_likelihood.item()



def mcmc_map_search(pairwise_prefs, demo_cnts, num_steps, step_stdev, confidence, normalize):
    '''run metropolis hastings MCMC and record weights in chain'''





    last_layer = np.random.randn(len(demo_cnts[0]))

    #normalize the weight vector to have unit 3-norm
    if normalize:
        last_layer = last_layer / np.linalg.norm(last_layer)
    #last_layer = euclidean_proj_l1ball(last_layer)

    #import time
    #start_t = time.time()
    #starting_loglik = calc_pairwise_ranking_loss(reward_net, demo_pairs, preference_labels)
    #end_t = time.time()
    #print("slow likelihood", starting_loglik, "time", 1000*(end_t - start_t))
    #start_t = time.time()
    starting_loglik = calc_linearized_pairwise_ranking_loss(last_layer, pairwise_prefs, demo_cnts, confidence)
    #end_t = time.time()
    #print("new fast? likelihood", new_starting_loglik, "time", 1000*(end_t - start_t))
    #print(bunnY)

    map_loglik = starting_loglik
    map_reward = copy.deepcopy(last_layer)

    cur_reward = copy.deepcopy(last_layer)
    cur_loglik = starting_loglik

    #print(cur_reward)

    reject_cnt = 0
    accept_cnt = 0
    chain = []
    log_liks = []
    for i in range(num_steps):
        #take a proposal step
        proposal_reward = cur_reward + np.random.normal(size=cur_reward.shape) * step_stdev #add random noise to weights of last layer

        #project
        if normalize:
            proposal_reward = proposal_reward / np.linalg.norm(proposal_reward)

        #calculate prob of proposal
        prop_loglik = calc_linearized_pairwise_ranking_loss(proposal_reward, pairwise_prefs, demo_cnts, confidence)
        if prop_loglik > cur_loglik:
            #accept always
            accept_cnt += 1
            cur_reward = copy.deepcopy(proposal_reward)
            cur_loglik = prop_loglik

            #check if this is best so far
            if prop_loglik > map_loglik:
                map_loglik = prop_loglik
                map_reward = copy.deepcopy(proposal_reward)
                print()
                print("step", i)

                print("updating map loglikelihood to ", prop_loglik)
                print("MAP reward so far", map_reward)
        else:
            #accept with prob exp(prop_loglik - cur_loglik)
            if np.random.rand() < np.exp(prop_loglik - cur_loglik):
                accept_cnt += 1
                cur_reward = copy.deepcopy(proposal_reward)
                cur_loglik = prop_loglik
            else:
                #reject and stick with cur_reward
                reject_cnt += 1
        chain.append(cur_reward)
        log_liks.append(cur_loglik)

    print("num rejects", reject_cnt)
    print("num accepts", accept_cnt)
    return map_reward, np.array(chain), np.array(log_liks)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--num_mcmc_steps', default=20000, type = int, help="number of proposals to generate for MCMC")
    parser.add_argument('--mcmc_step_size', default = 0.5, type=float, help="proposal step is gaussian with zero mean and mcmc_step_size stdev")
    parser.add_argument('--confidence', default=1, type=int, help='confidence in rankings, the beta parameter in the softmax')
    parser.add_argument('--normalize', default=False, action='store_true', help='turns on normalization so the reward function weight vectors have norm of 1')
    parser.add_argument('--features_dir', default='feature_files')

    args = parser.parse_args()

    ##NN files within PointBot_Networks to load into B-REX
    """
    model_files = ['PointBotGrid_alpha_0.92_lambda_0.12_vflr_0.01_pilr_0.01_2020_12_08.txt',
                'PointBotGrid_alpha_0.95_lambda_0_vflr_0.01_pilr_0.01_2020_12_07.txt',
                'PointBotGrid_alpha_0.95_lambda_0_vflr_0.01_pilr_0.001_2020_12_07.txt',
                'PointBotGrid_alpha_0.95_lambda_0.04_vflr_0.01_pilr_0.01_2020_12_07.txt',
                'PointBotGrid_alpha_0.95_lambda_0.2_vflr_0.001_pilr_0.001_2020_12_08.txt',
                'PointBotGrid_alpha_0_lambda_1_vflr_0.0001_pilr_0.01_2020_12_12.txt']
    """
    pickle_files = []
    import os
    for file_dir in ["five_demos"]:
        for filename in os.listdir(file_dir):
            if filename.endswith(".pkl"):
                pickle_files.append(file_dir + "/" + filename)
    print(pickle_files)

    demo_fcnts = []
    pairwise_prefs = []
    counter = 0
    for file in pickle_files:
        with open(file, 'rb') as handle:
            b = pickle.load(handle)
            demo_fcnts += [b["Bad"]]
            demo_fcnts += [b["Good"]]
            pairwise_prefs += [(counter, counter+1)]
            counter += 2

    print("Hand crafted feature expectations")
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


    #TODO: need to manually specify feature count vectors and true weights below

    # num_features = 3

    # demo_fcnts = np.array([[  0.0, 1.0,   0.1],
    #                       [  0.2, 0.5,   0.2],
    #                       [  0.8, 0.3,   0.3]])

    # true_weights = np.array([+1,-1,+1])

    num_features = 3 #inside obstacle, outside obstacle, number of trash
    """
    demo_fcnts = np.array([[0.0, 12.0],  #0: good trajectory that only goes in a little bit
                            [5.0, 8.0],  #1: bad trajectory that goes in a lot
                            [1.0, 20.0], #2: good traj that goes diretly to goal and avoids bad region
                            [0.0, 3.0]]) #3: good straight
    """

    #demo_fcnts = []
    #for file in model_files:
    #    demo_fcnts.append(model_eval(file, file[13:-15]))
    demo_fcnts = np.array(demo_fcnts).astype(float)

    #true pref ranking: 1,2,0,3
    #true_weights = np.array([-0.99, -.01]) #Currently only has 2 features
    #true_weights = np.array([-1, -.01, 3])

    #traj_returns = np.dot(demo_fcnts, true_weights)
    #print("returns", traj_returns)

    #pairwise_prefs =  [(1,0), (1,2), (2,0), (0,3), (2,3)]




    #just need index tuples (i,j) denoting j is preferred to i. Assuming all pairwise prefs for now
    #check if really better, there might be ties!
    """
    pairwise_prefs = []
    for i in range(len(traj_returns)):
        for j in range(i+1, len(traj_returns)):
            if traj_returns[i] < traj_returns[j]:
                pairwise_prefs.append((i,j))
            elif traj_returns[i] > traj_returns[j]:
                pairwise_prefs.append((j,i))
            else: # they are equal
                print("equal prefs", i, j, traj_returns[i], traj_returns[j])
                pairwise_prefs.append((i,j))
                pairwise_prefs.append((j,i))
    print("pairwise prefs (i,j) where i < j")
    print(pairwise_prefs)
    """

    #Run mcmc
    best_reward_lastlayer,chain,log_liks = mcmc_map_search(pairwise_prefs, demo_fcnts, args.num_mcmc_steps, args.mcmc_step_size, args.confidence, args.normalize)

    print("best reward weights", best_reward_lastlayer)
    print("predictions over ranked demos vs. ground truth returns")
    pred_returns = np.dot(demo_fcnts, best_reward_lastlayer)
    #print(pred_returns)
    #for i, p in enumerate(pred_returns):
     #   print(i,p,traj_returns[i])

    print("average reward weights")
    print(np.mean(chain, axis=0))


    print("10 random chain samples")
    for _ in range(10):
        w = random.choice(chain)
        print(w)


    """
    #Start saving reward distribution
    sampling_rate = 400
    sampling_chain = chain[100:]
    bounds = ((-90, 0), (-90, 60), (0, 90))
    box_size = 30

    num_x_tallies = int ((bounds[0][1] - bounds[0][0]) / float(box_size))
    num_y_tallies = int ((bounds[1][1] - bounds[1][0]) / float(box_size))
    num_z_tallies = int ((bounds[2][1] - bounds[2][0]) / float(box_size))

    tally = np.zeros((num_x_tallies, num_y_tallies, num_z_tallies))
    for i in range(sampling_chain.shape[0]):
        w = sampling_chain[i]

        x = abs((w[0] - bounds[0][0]) // box_size)
        y = abs((w[1] - bounds[1][0]) // box_size)
        z = abs((w[2] - bounds[2][0]) // box_size)

        if (y == 6):
            print('y:' + str(y))
        if (i%sampling_rate == 0) and (0 <= x < tally.shape[0]) and (0 <= y < tally.shape[1]) and (0 <= z < tally.shape[2]):
            tally[int(x), int(y), int(z)] += 1
    print(tally.shape[0])
    print(tally.shape[1])
    print(tally.shape[2])
    print(tally)
    tally = tally / float(np.sum(tally))
    weight_to_prob = {}


    for i in range(tally.shape[0]):
        for j in range(tally.shape[1]):
            for k in range(tally.shape[2]):
                if tally[i, j, k] > 0:
                    weight_to_prob[(bounds[0][0]+i * box_size + (box_size/2.0), bounds[1][0]+j * box_size + (box_size/2.0), bounds[2][0]+k * box_size + (box_size/2.0))] = tally[i, j, k]

    print('WEIGHTS TO PROBABILITY')
    print(weight_to_prob)

    with open('brex_reward_dist_unnormalized.pickle', 'wb') as handle:
        pickle.dump(weight_to_prob , handle, protocol=pickle.HIGHEST_PROTOCOL)
    #End saving reward distribution
    """


    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(log_liks)
    plt.title("log likelihoods")
    plt.figure()
    for i in range(num_features):
            plt.plot(chain[:,i],label='feature ' + str(i))
    plt.title("features")
    plt.legend()
    plt.show()
