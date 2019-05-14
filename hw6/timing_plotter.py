#
# Plot the results from the timings.
#
# Author: Terrence Alsup

from matplotlib import pyplot as plt

j = [0, 1, 2, 3]
cores = [1, 4, 16, 64]
N = [400, 800, 1600, 3200]

# Done with 10^4 iterations of the Jacobi 2D smoother on Prince.
weak_scaling_times = [2.33, 2.55, 2.58, 2.73]


strong_scaling_times = [15.73, 3.60, 1.22, 0.43]
s = strong_scaling_times[0]
ideal_scaling = [s, s/4, s/16, s/64]

plt.figure(1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.semilogx(cores, weak_scaling_times, 'b-x', lw = 2)
plt.xticks(cores, cores)
plt.xlabel('Number of Processors: $P$')
plt.ylabel('Runtime [s]')
plt.title('Weak Scaling for 2D Jacobi Smoother')

plt.figure(2)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.loglog(cores, strong_scaling_times, 'b-x', lw = 2, label='Empirial Scaling')
plt.loglog(cores, ideal_scaling, 'r--', lw = 2, label = 'Ideal Scaling')
plt.xticks(cores, cores)
plt.xlabel('Number of Processors: $P$')
plt.ylabel('Runtime [s]')
plt.title('Strong Scaling for 2D Jacobi Smoother')
plt.legend()



plt.show()
