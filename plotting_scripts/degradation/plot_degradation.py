import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--results_file', type=str, default='cartpole20vscartpole2', help='file for plotting that is in results/degradation')

args = parser.parse_args()


results = pickle.load( open( "results/degradation/{}.p".format(args.results_file), "rb" ) )



x = range(0,11)
ave_perf = np.mean(results, axis=0)
std_perf = np.std(results, axis=0)
plt.xlabel('num bad demos')
plt.ylabel('ave performance of bc policy')
plt.fill_between(x, ave_perf - std_perf , ave_perf + std_perf, facecolor='green', alpha=0.5)
plt.plot(x, ave_perf, 'g')
plt.title(args.results_file)
plt.savefig("plotting_scripts/degradation/" + args.results_file + ".png")
plt.show()    
