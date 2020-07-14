import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

NUMPES = 2 # number of potential energy surfaces

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

# plot preparation
x, p, t = read_input()
LEN_X = len(x)
LEN_P = len(p)
LEN_T = len(t)
dx = (x[LEN_X-1] - x[0]) / (LEN_X - 1)
dp = (p[LEN_P-1] - p[0]) / (LEN_P - 1)
xv, pv = np.meshgrid(x, p) # transform to vector for plotting

# plot population evolution
def calc_pop():
	file = open('phase.txt', 'r')
	ppl = [[],[]]
	for i in range(LEN_T):
		ppl[0].append(sum(np.array(file.readline().split(), dtype=float)) * dx * dp)
		file.readline() # rho[0][1]
		file.readline() # rho[1][0]
		ppl[1].append(sum(np.array(file.readline().split(), dtype=float)) * dx * dp)
		file.readline() # blank line
	file.close()
	return ppl

ppl = calc_pop()
plt.plot(t, ppl[0], color='r', label='Population[0]')
plt.plot(t, ppl[1], color='b', label='Population[1]')
plt.legend(loc = 'best')
plt.xlim((t[0],t[LEN_T-1]))
plt.ylim((0,1))
plt.savefig('phase.png')


plt.clf()
# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
xmax = [[0.2, 0.1], [0.1, 0.05]]
levels = []
for i in range(NUMPES):
	levels.append([])
	for j in range(NUMPES):
		levels[i].append(MaxNLocator(nbins=15).tick_values(-xmax[i][j],xmax[i][j])) # color region
cmap = plt.get_cmap('seismic') # the kind of color: red-white-blue
#norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True) # the mapping rule

# initialize the plot
fig, axs = plt.subplots(nrows=NUMPES, ncols=NUMPES, figsize=(20,10))

# set time text
time_template = 'time = %da.u.'

# initialization: blank page
def init():
	# clear, then set x/y label, title of subplot, and colorbar
	cfs = []
	for i in range(NUMPES):
		cfs.append([])
		for j in range(NUMPES):
			axs[i][j].clear()
			axs[i][j].set_xlabel('x')
			axs[i][j].set_ylabel('p')
			if i == j:
				axs[i][j].set_title(r'$\rho_{%d%d}$' % (i, j))
			elif i < j:
				axs[i][j].set_title(r'$\Re(\rho_{%d%d})$' % (i, j))
			else:
				axs[i][j].set_title(r'$\Im(\rho_{%d%d})$' % (j, i))
			cfs[i].append(axs[i][j].contourf(xv, pv, np.zeros((LEN_X,LEN_P)), levels=levels[i][j], cmap=cmap))
			fig.colorbar(cfs[i][j], extend='both', ax=axs[i][j])
			cfs[i][j].set_clim(-xmax[i][j], xmax[i][j])
	# figure settings: make them closer, title to be time
	fig.suptitle('')
	return fig, axs,

# animation: each timestep in phase.txt
def ani(i):
	# get data, in rhoi_real
	file = open('phase.txt', 'r')
	# old data
	for j in range(i - 1):
		for k in range(NUMPES*NUMPES):
			file.readline() # rho
		file.readline() # blank line
	# new data
	rho = []
	for j in range(NUMPES):
		rho.append([])
		for k in range(NUMPES):
			rho[j].append(np.array(file.readline().split(), dtype=float))
	LENGTH = len(rho[0][0])//2
	for j in range(LENGTH):
		for k in range(NUMPES):
			rho[k][k][j] = rho[k][k][2*j]
			for l in range(k+1,NUMPES):
				rho[k][l][j] = (rho[k][l][2*j]+rho[l][k][2*j])/2.0
				rho[l][k][j] = (rho[k][l][2*j+1]-rho[l][k][2*j+1])/2.0
	for j in range(NUMPES):
		for k in range(NUMPES):
			rho[j][k] = rho[j][k][:LENGTH].reshape(LEN_P,LEN_X).T
	file.close()

	# print contourfs
	for j in range(NUMPES):
		for k in range(NUMPES):
			axs[j][k].contourf(xv, pv, rho[j][k], levels=levels[j][k], cmap=cmap)
	fig.suptitle(time_template % t[i])
	return fig, axs,

# make the animation
ani = animation.FuncAnimation(fig, ani, LEN_T, init, interval=10000//(t[1]-t[0]), repeat=False, blit=False)
# show
ani.save('phase.gif','imagemagick')
# plt.show()
