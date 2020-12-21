from matplotlib import colors
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator

NUMPES = 2 # number of potential energy surfaces
FIGSIZE = 10 # length of one figure
NOPARAM = 1 + 1 + NUMPES * (NUMPES + 1) // 2 # number of hyperparameters: noise, and gaussian ARD

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

# plot -ln(likelihood)
index = 0
plt.plot(t, data[index], label='-log(marg_ll)')
plt.xlabel('t/a.u.')
plt.xlim((t[0], t[LEN_T-1]))
plt.ylabel('-ln(likelihood)')
plt.title('Negative Log Marginal Likelihood')
plt.savefig('marg_ll.png')
plt.clf()
index += 1

 # plot hyperparameters
hyperparameters = data[index:index+NUMPES*NUMPES*NOPARAM].reshape(NUMPES, NUMPES, NOPARAM, LEN_T)
for i in range(NUMPES):
	for j in range(NUMPES):
		fig, ax1 = plt.subplots()
		ax1.semilogy(t, hyperparameters[i][j][0], label='Diagonal Weight')
		ax1.semilogy(t, hyperparameters[i][j][1], label='Gaussian Weight')
		idx = 2;
		ax2 = ax1.twinx() # for characteristic length
		ax3 = ax1.twinx() # for cross-terms
		ax3.spines["right"].set_position(("axes", 1.2))
		make_patch_spines_invisible(ax3)
		ax3.spines["right"].set_visible(True)
		prop_cycler = ax1._get_lines.prop_cycler
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
		ax1.set_xlim((t[0], t[LEN_T-1]))
		ax1.set_xlabel('t/a.u.')
		ax1.tick_params('y', colors='red')
		ax1.set_ylabel('Weights', color='red')
		ax2.tick_params('y', colors='blue')
		ax2.set_ylabel('Characteristic Lengths', color='blue')
		ax3.tick_params('y', colors='green')
		ax3.set_ylabel('Cross Term', color='green')
		ax1.set_title('Hyperparameters of ' + element_name[i][j])
		lines1, labels1 = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		lines3, labels3 = ax3.get_legend_handles_labels()
		ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='best')
		fig.savefig('rho%d%d'%(i,j)+'_param.png', bbox_inches='tight')
		plt.clf()
index += NUMPES * NUMPES * NOPARAM

# plot MSE
mse = data[index:index+NUMPES*NUMPES*2].reshape(NUMPES, NUMPES, 2, LEN_T)
for i in range(NUMPES):
	for j in range(NUMPES):
		plt.semilogy(t, mse[i][j][0], label=element_name[i][j])
		plt.semilogy(t, mse[i][j][1], label='Constrained '+element_name[i][j])
plt.xlim((t[0], t[LEN_T-1]))
plt.xlabel('t/a.u.')
plt.ylabel('lg(MSE)')
plt.title('Mean Square Error')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('mse.png', bbox_inches='tight')
plt.clf()
index += NUMPES * NUMPES * 2

# plot normalization and energy conservation
norm_and_energy = data[index:].reshape(NUMPES, 3, 5, LEN_T)
norm_and_energy_sum_over_pes=norm_and_energy.sum(axis=0)
second_dim_name = ['Norm', 'Potential Energy', 'Kinetic Energy']
second_dim_file_name = ['norm', 'potential', 'kinetic']
third_dim_name = ['Exact Solution', 'Grid without Constraint', 'Params without Constraint', 'Grid', 'Params']
third_dim_file_name = ['exact', 'grid_no_constraint', 'params_no_constraint', 'grid', 'params']
# plot norm, potential energy and kinetic energy
for i in range(3):
	for j in range(5):
		for k in range(NUMPES):
			plt.plot(t, norm_and_energy[k][i][j], label=element_name[k][k]+' '+second_dim_name[i]+' of '+third_dim_name[j])
		plt.plot(t, norm_and_energy_sum_over_pes[i][j], label='Total '+second_dim_name[i]+' of '+third_dim_name[j])
		plt.xlim((t[0], t[LEN_T-1]))
		plt.xlabel('t/a.u.')
		if i == 0:
			plt.ylim((0.0, 1.0))
			plt.ylabel(second_dim_name[i])
		else:
			plt.ylabel(second_dim_name[i]+'/a.u.')
		plt.title(second_dim_name[i])
		plt.legend()
		plt.savefig(second_dim_file_name[i]+'_'+third_dim_file_name[j]+'.png')
		plt.clf()
# plot total energy
for j in range(5):
	for k in range(NUMPES):
		plt.plot(t, norm_and_energy[k][1][j]+norm_and_energy[k][2][j], label=element_name[k][k]+' Total Energy of '+third_dim_name[j])
	plt.plot(t, norm_and_energy_sum_over_pes[1][j]+norm_and_energy_sum_over_pes[2][j], label='Total Energy of '+third_dim_name[j])
	plt.xlim((t[0], t[LEN_T-1]))
	plt.xlabel('t/a.u.')
	plt.ylabel('Total Energy/a.u.')
	plt.title('Total Energy')
	plt.legend()
	plt.savefig('total'+'_'+third_dim_file_name[j]+'.png')
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
