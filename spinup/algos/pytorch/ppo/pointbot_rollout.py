import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym
import time
from tqdm import tqdm
import os, sys
from json import JSONEncoder
import json
import spinup.algos.pytorch.vpg.core as core
from spinup.utils.logx import EpochLogger
from spinup.envs.pointbot_const import *
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.examples.pytorch.broil_rtg_pg_v2.pointbot_reward_utils import PointBotReward
from spinup.examples.pytorch.broil_rtg_pg_v2.cartpole_reward_utils import CartPoleReward
# from spinup.examples.pytorch.broil_rtg_pg_v2.cheetah_reward_utils import CheetahReward
from spinup.examples.pytorch.broil_rtg_pg_v2.reacher_reward_utils import ReacherReward
from spinup.examples.pytorch.broil_rtg_pg_v2.shelf_reward_utils import ShelfReward
from spinup.examples.pytorch.broil_rtg_pg_v2.cvar_utils import cvar_enumerate_pg
import dmc2gym
torchify = lambda x: torch.FloatTensor(x).to(torch.device('cpu'))


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    #rew_dim is the dimensionality of the reward function posterior
    def __init__(self, obs_dim, act_dim, num_rew_fns, size, gamma=0.99, lam=0.95):
        self.num_rew_fns = num_rew_fns
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(core.combined_shape(size, num_rew_fns), dtype=np.float32)
        self.rew_buf = np.zeros(core.combined_shape(size, num_rew_fns), dtype=np.float32)
        self.ret_buf = np.zeros(core.combined_shape(size, num_rew_fns), dtype=np.float32)
        self.val_buf = np.zeros(core.combined_shape(size, num_rew_fns), dtype=np.float32)
        self.posterior_returns = []
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=None):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        if last_val is None:
            last_val = np.zeros(self.num_rew_fns, dtype=np.float32)

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.vstack((self.rew_buf[path_slice], last_val))
        vals = np.vstack((self.val_buf[path_slice], last_val))

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        #TODO: see if there is a way to vectorize this
        for i in range(self.num_rew_fns):
            self.adv_buf[path_slice,i] = core.discount_cumsum(deltas[:,i], self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        # also store the cumulative returns for BROIL CVaR calculation
        self.posterior_returns.append(np.sum(rews, axis=0))

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        #TODO: see if we can vectorize this and figure out multithreading
        for i in range(self.num_rew_fns):
            adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf[:,i])
            self.adv_buf[:,i] = (self.adv_buf[:,i] - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                        adv=self.adv_buf, logp=self.logp_buf, p_returns=self.posterior_returns)
        self.posterior_returns = []
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def ppo(env_fn, reward_dist, broil_risk_metric='cvar', actor_critic=core.BROILActorCritic, ac_kwargs=dict(), render=False, seed=0,
        steps_per_epoch=4000, epochs=50, broil_lambda=0.5, broil_alpha=0.95, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters =40, train_v_iters=80, lam=0.97, max_ep_len=1000, target_kl = .01,
        clip_ratio = .2, logger_kwargs=dict(), save_freq=10, clone=False, num_demos=0, train_pi_BC_iters=100):
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================
            The ``act`` method behaves the same as ``step`` but only returns ``a``.
            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================
            The ``v`` module's forward call should accept a batch of observations
            and return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.
        broil_lambda (float): amount to blend between maximizing expected return (1.0)
            and maximizing CVaR (0.0). Always between 0 and 1.
        broil_alpha (float): risk sensitivity in range [0,1) for computing alpha-CVaR
            higher alpha is more risk sensitive.
        gamma (float): Discount factor. (Always between 0 and 1.)
        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.
        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.


    def get_reward_distribution(args, reward_dist, env, next_o, action):
        if args.env == 'CartPole-v0':
            rew_dist = reward_dist.get_reward_distribution(next_o)
        elif args.env == 'PointBot-v0':
            rew_dist = reward_dist.get_reward_distribution(env, next_o)
        else:
            raise NotImplementedError("Unsupported Environment")

        return rew_dist

    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create BROIL actor-critic module
    num_rew_fns = reward_dist.posterior.size

    checkpoint = torch.load("PointBot-v0_grid_alpha_0.95_lambda_0.2_vflr_0.001_pilr_0.0003_2021_01_15.txt")
    ac = checkpoint['model']
    ac.load_state_dict(checkpoint['state_dict'])
    for parameter in ac.parameters():
        parameter.requires_grad = False

    ac.eval()

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, num_rew_fns, local_steps_per_epoch, gamma, lam)

    mean_r = torch.zeros(num_rew_fns)
    std_r = torch.zeros(num_rew_fns)
    if clone:
        demo_obs, demo_acs = env.get_demos(num_demos)


    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    #TODO: see if we can get away with one adam optimizer for family of networks...
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    #vf_optimizers = [Adam(ac.v.v_nets[i].parameters(), lr=vf_lr) for i in range(num_rew_fns)]

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    cvar_list = []
    wc_ret_list = []
    bc_ret_list = []
    ret_list = []
    obstacle_list = []
    trajectories_x = []
    trajectories_y = []
    trash_trajectories = []
    num_trashes = []
    obs_times = []

    # Main loop: collect experience in env and update/log each epoch
    for epoch in tqdm(range(rollouts)):
        first_rollout = True
        running_cvar = []
        total_reward_dist = np.zeros(num_rew_fns)
        running_ret = 0
        num_runs = 0
        obstacles = 0
        num_constraint_violations = 0
        num_episodes = 0
        constraint_violated = False
        for t in tqdm(range(local_steps_per_epoch)):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            # TODO: Test unnormalizing the values
            v = (v * std_r.numpy()) + mean_r.numpy()

            next_o, r, d, info = env.step(a)
            if args.env == 'Shelf-v0' or args.env == 'reacher': # Check if you ever violate a constraint in this episode
                if info['constraint']:
                    constraint_violated = True
            #TODO: check this, but I think reward as function of next state makes most sense
            rew_dist = get_reward_distribution(args, reward_dist, env, next_o, a)
            total_reward_dist += rew_dist.flatten()
            running_ret += r
            ep_ret += r
            ep_len += 1
            if args.env == 'PointBot-v0':
                obstacles += int(env.obstacle(next_o))



            # save and log
            buf.store(o, a, rew_dist, v, logp)
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if render and first_rollout:
                env.render()
                time.sleep(0.01)

            if terminal or epoch_ended:
                if constraint_violated:
                    num_constraint_violations += 1
                num_episodes += 1

                constraint_violated = False
                first_rollout = False
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                    buf.finish_path(v)
                else:
                    buf.finish_path()

                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    # print(ep_ret)

                num_runs += 1

                if args.env == 'PointBot-v0':
                    last_trajectory = np.array(env.hist)
                    obs_times.append(env.obs_time/100)
                    if TRASH:
                        num_trashes.append(len(env.current_trash_taken))
                    if epoch == epochs - 1:
                        trajectories_x.append(last_trajectory[:, 0])
                        trajectories_y.append(last_trajectory[:, 2])
                        if TRASH:
                            env.current_trash_taken.append(env.next_trash)
                            trash_trajectories.append(env.current_trash_taken)

                o, ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Store stuff for saving data
        ret_list.append(running_ret / float(num_runs))
        wc_ret_list.append(np.min(total_reward_dist) / float(num_runs))
        bc_ret_list.append(np.max(total_reward_dist / float(num_runs)))
        cvar_list.append(sum(running_cvar))
        obstacle_list.append(obstacles / float(num_runs))
        """print('True returns:', ret_list)
        print('Cvar: ', cvar_list)
        print('Worst case:', wc_ret_list)"""


        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Risk', average_only=True)
        logger.log_tabular('ExpectedRet', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
        print("Frac Constraint Violations: %d/%d" % (num_constraint_violations, num_episodes))


    file_data = 'broil_data_109/'
    experiment_name = args.env + '_alpha_' + str(broil_alpha) + '_lambda_' + str(broil_lambda) + 'rollout'

    metrics = {"conditional value at risk": ('_cvar', cvar_list),
               "true_return": ('_true_return', ret_list),
               "worst case return": ('_worst_case_return', wc_ret_list),
               "best case return": ('_best_case_return', bc_ret_list),
               "obstacle_collision": ('_obstacles', obstacle_list)}

    for metric, result in metrics.items():
        file_metric_description, results = result
        file_path = file_data + 'results/' + experiment_name + file_metric_description + '.txt'
        #assert not os.path.isfile(file_path)  # make sure we are making a new file and not overwriting
        with open(file_path, 'w') as f:
            for item in results:
                f.write("%s\n" % item)

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    if args.env == 'PointBot-v0':
        plt.ylim((env.grid[2], env.grid[3]))
        plt.xlim((env.grid[0], env.grid[1]))
        for i in range(4):
            x = trajectories_x[i]
            y = trajectories_y[i]
            if i == 0 and args.combiner:
                decodeddict = None
                if (os.path.exists(file_data + "states_combined.txt")):
                    f = open(file_data + "states_combined.txt", "r")
                    decodeddict = json.load(f)
                    f.close()
                a, b = str(broil_lambda) + "_x", str(broil_lambda) + "_y"
                temp = {a: trajectories_x, b: trajectories_y}
                if (os.path.exists(file_data + "states_combined.txt")):
                    os.remove(file_data + "states_combined.txt")
                f = open(file_data + "states_combined.txt", "w")
                if decodeddict == None:
                    json.dump(temp, f, cls=NumpyArrayEncoder)
                else:
                    decodeddict.update(temp)
                    json.dump(decodeddict, f, cls=NumpyArrayEncoder)
                f.close()

            plt.scatter([x[0]],[y[0]],  [6], '#00FF00', zorder=11)
            plt.scatter(x[1:], y[1:], len(x)*[6], zorder=9)
            if TRASH:
                for j in trash_trajectories[i]:
                    plt.scatter([j[0]],[j[1]], [25], zorder = 10, color = '#000000')

        x_bounds = [obstacle.boundsx for obstacle in env.obstacle.obs]
        y_bounds = [obstacle.boundsy for obstacle in env.obstacle.obs]
        for i in range(len(x_bounds)):
            plt.gca().add_patch(patches.Rectangle((x_bounds[i][0], y_bounds[i][0]), width=x_bounds[i][1] - x_bounds[i][0], height=y_bounds[i][1] - y_bounds[i][0], fill=True, alpha=.5, linewidth=1, zorder = 0, edgecolor='#d3d3d3',facecolor='#d3d3d3'))
        
        plt.savefig(file_data + 'visualizations/' + experiment_name + '.png')
        plt.clf()
        torch.save(ac.state_dict(), file_data + 'PointBot_networks/' + experiment_name + '.pt')

    print(' Data from experiment: ', experiment_name, ' saved.')
    if args.env == 'PointBot-v0':
        print(np.average(num_trashes[-100:]))
        print(np.std(num_trashes[-100:]))
        print(np.average(obs_times[-100:]))
        print(np.std(obs_times[-100:]))

    print(num_runs)

if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--policy_lr', type=float, default=3e-4, help="learning rate for policy")
    parser.add_argument('--value_lr', type=float, default=1e-3)
    parser.add_argument('--risk_metric', type=str, default='cvar', help='choice of risk metric, options are "cvar" or "erm"' )
    parser.add_argument('--broil_lambda', type=float, default=1, help="blending between cvar and expret")
    parser.add_argument('--broil_alpha', type=float, default=0.95, help="risk sensitivity for cvar")
    parser.add_argument('--clone', action="store_true", help="do behavior cloning")
    parser.add_argument('--num_demos', type=int, default=0)
    parser.add_argument('--combiner', type=bool, default=True)
    parser.add_argument('--rollouts', type=bool, default=True)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    if args.env == 'CartPole-v0':
        reward_dist = CartPoleReward()
    elif args.env == 'PointBot-v0':
        reward_dist = PointBotReward()
    else:
        raise NotImplementedError("Unsupported Environment")

    if args.env == 'reacher':
        env_fn = lambda: dmc2gym.make(domain_name='reacher', task_name='hard', seed=args.seed)
    elif args.env == 'manipulator':
        env_fn = lambda: dmc2gym.make(domain_name='manipulator', task_name='bring_ball', seed=args.seed)
    else:
        env_fn = lambda : gym.make(args.env)

    ppo(env_fn, reward_dist=reward_dist, broil_risk_metric=args.risk_metric, broil_lambda=args.broil_lambda, broil_alpha=args.broil_alpha,
        actor_critic=core.BROILActorCritic, render=args.render,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        pi_lr=args.policy_lr, vf_lr = args.value_lr, logger_kwargs=logger_kwargs, clone=args.clone, num_demos=args.num_demos)