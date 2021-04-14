# Policy Gradient Bayesian Robust Optimization for Imitation Learning

## Installation

Dependencies for the CartPole environment can be installed by installing Open AI Gym. Dependencies for the Reacher environment can be installed by pip installing the modified version of the dm_control package and an Open AI gym wrapper for the package which are both included in the included source. 

This code repo builds on the [OpenAI Spinning Up gitrepo](https://spinningup.openai.com/en/latest/user/installation.html). First follow the instructions to install:

first clone the repo then

```
cd bcprefs
conda env create -f environment.yml
conda activate bcprefs
pip install -e .
```

Optional: install dm_control suite
```

pip install dm_control dmc2gym
```

To run RL to train a demonstration (replace env with other env if desired and change number of epochs to control how optimal it is (lower is less optimal))
```
python -m spinup.run ppo_pytorch --env CartPole-v0 --epochs 20 --exp_name cartpole20

```

The data gets written to a data folder

You can then generate demos from the trained policy for use in behavioral cloning:
```
python spinup/utils/generate_demos.py --load_path data/cartpole20/cartpole20_s0/ --save_path cartpole20
```

This example loads from the pretrained policy and by default will save 10 demos to a pickle file

You can now run BC on it
```
python spinup/algos/pytorch/bc/bc.py --load_path demonstrations/cartpole20.p --env CartPole-v0
```


You can generate data for BC degradation if you pre-train two RL policies, generate demos from them both, and then give paths to the demos from both policies labeled as good and bad demos. For example:
```
 python spinup/experiments/bc_prefs/bad_demo_degradation.py --good_load_path demonstrations/cartpole10.p --bad_load_path demonstrations/cartpole2.p --env CartPole-v0 --save_path results/degradation/

```
here I've used data from a 10 epoch RL algo (near optimal) and a 2 epoch one (bad)

This will save to ```results/degradation/cartpol120vscarpole2.p```

You can then plot using
```
python plotting_scripts/degradation/plot_degradation.py --results_file cartpole10vscartpole2
```

You can also try and learn from the good and bad by telling the policy to maximize good and maximize diff between good and bad to hopefuly avoid the bad actions:

```
python spinup/experiments/bc_prefs/good_good_minus_bad_demo_degradation.py --good_load_path demonstrations/cartpole20.p --bad_load_path demonstrations/cartpole2.p --env CartPole-v0 --save_path results/degradation/
```

It doesn't work too well when using 10epoch RL as good and 2-epoch RL as bad. You can compare performance with 

```
python plotting_scripts/degradation/plot_comparison_degradation.py --results_file cartpole10vscartpole2
```




