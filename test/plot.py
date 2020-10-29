import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator

NUMPES = 2 # number of potential energy surfaces
FIGSIZE = 10 # length of one figure

# read input grid and time
def read_input():
	xfile = open('x.txt', 'r')
	x = np.array(xfile.readlines(), dtype=float)
	xfile.close()
	pfile = open('p.txt', 'r')
	p = np.array(pfile.readlines(), dtype=float)
	pfile.close()
	tfile = open('t.txt', 'r')
	t = np.array(tfile.readlines(), dtype=int)
	tfile.close()
	return x, p, t

# read MSE, -log(likelihood) and hyperparameters for plotting
def read_log():
	file = open('log.txt', 'r')
	data = []
	for line in file.readlines():
		data.append(np.array(line.split(), dtype=float)[1:])
	file.close()
	return data

# make the whole frame invisible
def make_patch_spines_invisible(ax):
	ax.set_frame_on(True)
	ax.patch.set_visible(False)
	for sp in ax.spines.values():
		sp.set_visible(False)


# preparation for plot
x, p, t = read_input()
LEN_X, LEN_P, LEN_T = len(x), len(p), len(t)
dx = (x[LEN_X-1] - x[0]) / (LEN_X - 1)
dp = (p[LEN_P-1] - p[0]) / (LEN_P - 1)
xv, pv = np.meshgrid(x, p) # transform to vector for plotting
element_name = [] # name of each element of PWTDM
for i in range(NUMPES):
	element_name.append([])
	for j in range(NUMPES):
		if i == j:
			element_name[i].append(r'$\rho_{%d%d}$' % (i, j))
		elif i < j:
			element_name[i].append(r'$\Re(\rho_{%d%d})$' % (i, j))
		else:
			element_name[i].append(r'$\Im(\rho_{%d%d})$' % (j, i))
data = np.array(read_log()).T

# plot MSE and -ln(likelihood)
mse = data[:NUMPES*NUMPES][:].reshape(NUMPES, NUMPES, LEN_T)
# plot mse
fig, ax1 = plt.subplots()
for i in range(NUMPES):
	for j in range(NUMPES):
		ax1.semilogy(t, mse[i][j], label=element_name[i][j])
ax1.tick_params('y', colors='red')
ax1.set_ylabel('lg(MSE)', color='red')
# plt -ln(likelihood)
log_marg_ll = data[NUMPES*NUMPES][:]
ax2 = ax1.twinx()
ax2._get_lines.prop_cycler = ax1._get_lines.prop_cycler
ax2.plot(t, log_marg_ll, label='-log(marg_ll)')
ax2.tick_params('y', colors='blue')
ax2.set_ylabel('-ln(likelihood)', color='blue')
# set axis
ax1.set_xlim((t[0], t[LEN_T-1]))
ax1.set_xlabel('t/a.u.')
ax1.set_title('log10 Mean Square Error and Negative Log Marginal Likelihood')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
fig.savefig('mse_and_marg_ll.png')
plt.clf()

 # plot hyperparameters
NOPARAM = 1 + 1 + NUMPES * (NUMPES + 1) // 2 # number of hyperparameters: noise, and gaussian ARD
hyperparameters = data[NUMPES*NUMPES+1:][:].reshape(NUMPES, NUMPES, NOPARAM, LEN_T)
fig, ax1 = plt.subplots(nrows=NUMPES, ncols=NUMPES, figsize=(NUMPES*FIGSIZE*2, NUMPES*FIGSIZE))
for i in range(NUMPES):
	for j in range(NUMPES):
		ax1[i][j].semilogy(t, hyperparameters[i][j][0], label='Diagonal Weight')
		ax1[i][j].semilogy(t, hyperparameters[i][j][1], label='Gaussian Weight')
		idx = 2;
		ax2 = ax1[i][j].twinx() # for characteristic length
		ax3 = ax1[i][j].twinx() # for cross-terms
		ax3.spines["right"].set_position(("axes", 1.2))
		make_patch_spines_invisible(ax3)
		ax3.spines["right"].set_visible(True)
		prop_cycler = ax1[i][j]._get_lines.prop_cycler
		for k in range(NUMPES):
			for l in range(k, NUMPES):
				if (k == l):
					ax2._get_lines.prop_cycler = prop_cycler
					ax2.semilogy(t, 1.0 / hyperparameters[i][j][idx], label='Gaussian Characteristic Length %d' % k)
					prop_cycler = ax2._get_lines.prop_cycler
				else:
					ax3._get_lines.prop_cycler = prop_cycler
					ax3.plot(t, hyperparameters[i][j][idx], label='Gaussian Relevance between %d and %d' % (k, l))
					prop_cycler = ax3._get_lines.prop_cycler
				idx += 1
		ax1[i][j].set_xlim((t[0], t[LEN_T-1]))
		ax1[i][j].set_xlabel('t/a.u.')
		ax1[i][j].tick_params('y', colors='red')
		ax1[i][j].set_ylabel('Weights')
		ax2.tick_params('y', colors='blue')
		ax2.set_ylabel('Characteristic Lengths')
		ax3.tick_params('y', colors='green')
		ax3.set_ylabel('Cross Term')
		ax1[i][j].set_title('Hyperparameters of ' + element_name[i][j])
		lines1, labels1 = ax1[i][j].get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		lines3, labels3 = ax3.get_legend_handles_labels()
		ax1[i][j].legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='best')
fig.savefig('param.png')
plt.clf()

# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
xmax = [[0.4, 0.1], [0.1, 0.05]]
levels = []
for i in range(NUMPES):
	levels.append([])
	for j in range(NUMPES):
		levels[i].append(MaxNLocator(nbins=15).tick_values(-xmax[i][j],xmax[i][j])) # color region
cmap = plt.get_cmap('seismic') # the kind of color: red-white-blue

# initialize the plot
NROW, NCOL = NUMPES, NUMPES * 2
fig, axs = plt.subplots(nrows=NROW, ncols=NCOL, figsize=(NCOL*FIGSIZE*2,NROW*FIGSIZE))

# set time text
time_template = 'time = %da.u.'

# initialization: blank page
def init():
	# clear, then set x/y label, title of subplot, and colorbar
	cfs = []
	for i in range(NUMPES):
		cfs.append([])
		for j in range(NUMPES*2):
			axs[i][j].clear()
			# set title
			title = ''
			if j % 2 == 0:
				title += 'Actual'
			else:
				title += 'Simualted'
			title += ' ' + element_name[i][j//2]
			axs[i][j].set_title(title)
			# set other things
			axs[i][j].set_xlabel('x')
			axs[i][j].set_ylabel('p')
			cfs[i].append(axs[i][j].contourf(xv, pv, np.zeros((LEN_X,LEN_P)), levels=levels[i][j//2], cmap=cmap))
			fig.colorbar(cfs[i][j], extend='both', ax=axs[i][j])
			cfs[i][j].set_clim(-xmax[i][j//2], xmax[i][j//2])
	# figure settings: make them closer, title to be time
	fig.suptitle('')
	return fig, axs,

# animation: each timestep in phase.txt
def ani(i):
	# get data, in rhoi_real
	origin = open('phase.txt', 'r')
	sim = open('sim.txt', 'r')
	choose = open('choose.txt', 'r')
	# old data
	for j in range(i-1):
		for k in range(NUMPES*NUMPES):
			origin.readline() # actual rho
			sim.readline() # simulated rho
		origin.readline() # blank line
		sim.readline() # blank line
		choose.readline() # points
	# new data
	rho_orig = []
	rho_sim = []
	point = np.array(choose.readline().split(), dtype=float)
	for j in range(NUMPES):
		rho_orig.append([])
		rho_sim.append([])
		for k in range(NUMPES):
			rho_orig[j].append(np.array(origin.readline().split(), dtype=float))
			rho_sim[j].append(np.array(sim.readline().split(), dtype=float))
	origin.close()
	sim.close()
	choose.close()

	# adjust data to proper form
	LENGTH = len(rho_orig[0][0]) // 2
	NPOINT = len(point) // 2
	for j in range(LENGTH):
		for k in range(NUMPES):
			rho_orig[k][k][j] = rho_orig[k][k][2*j]
			for l in range(k+1,NUMPES):
				rho_orig[k][l][j] = (rho_orig[k][l][2*j]+rho_orig[l][k][2*j]) / 2.0
				rho_orig[l][k][j] = (rho_orig[k][l][2*j+1]-rho_orig[l][k][2*j+1]) / 2.0
	for j in range(NUMPES):
		for k in range(NUMPES):
			rho_orig[j][k] = rho_orig[j][k][:LENGTH].reshape(LEN_P,LEN_X).T
			rho_sim[j][k] = rho_sim[j][k].reshape(LEN_P,LEN_X).T
	point = point.reshape(NPOINT,2).T

	# print contourfs
	for j in range(NUMPES):
		for k in range(NUMPES):
			axs[j][2*k].contourf(xv, pv, rho_orig[j][k], levels=levels[j][k], cmap=cmap)
			axs[j][2*k].scatter(point[0], point[1], s=3, c='black')
			axs[j][2*k+1].contourf(xv, pv, rho_sim[j][k], levels=levels[j][k], cmap=cmap)
			axs[j][2*k+1].scatter(point[0], point[1], s=3, c='black')
	fig.suptitle(time_template % t[i])
	return fig, axs,

# make the animation
ani = FuncAnimation(fig, ani, LEN_T, init, interval=10000//(t[1]-t[0]), repeat=False, blit=False)
# show
ani.save('phase.gif','imagemagick')
# plt.show()
