import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--results_file', type=str, default='cartpole20vscartpole2', help='file for plotting that is in results/degradation')

args = parser.parse_args()


bc_results = pickle.load( open( "results/degradation/{}.p".format(args.results_file), "rb" ) )
#gb_results = pickle.load( open( "results/degradation/goodbad_{}.p".format(args.results_file), "rb" ) )
g2b_results = pickle.load( open( "results/degradation/good2bad_{}.p".format(args.results_file), "rb" ) )



x = range(0,11)
bc_ave_perf = np.mean(bc_results, axis=0)
bc_std_perf = np.std(bc_results, axis=0)

#gb_ave_perf = np.mean(gb_results, axis=0)
#gb_std_perf = np.std(gb_results, axis=0)

g2b_ave_perf = np.mean(g2b_results, axis=0)
g2b_std_perf = np.std(g2b_results, axis=0)

plt.plot([0, 10],[bc_ave_perf[0], bc_ave_perf[0]],'k:')

plt.xlabel('num bad demos')
plt.ylabel('ave performance of bc policy')
plt.fill_between(x, bc_ave_perf - bc_std_perf , bc_ave_perf + bc_std_perf, facecolor='green', alpha=0.3)
plt.plot(x, bc_ave_perf, 'g', label="bc")

#plt.fill_between(x, gb_ave_perf - gb_std_perf , gb_ave_perf + gb_std_perf, facecolor='blue', alpha=0.3)
#plt.plot(x, gb_ave_perf, 'b', label="good_bad")


plt.fill_between(x, g2b_ave_perf - g2b_std_perf , g2b_ave_perf + g2b_std_perf, facecolor='red', alpha=0.3)
plt.plot(x, g2b_ave_perf, 'r', label="good2_bad")


plt.legend()
plt.title(args.results_file)
plt.savefig("plotting_scripts/degradation/goodbad_" + args.results_file + ".png")
plt.show()    
