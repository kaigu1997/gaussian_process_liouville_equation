import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import MaxNLocator
from scipy import stats

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

# read MSE and -log(likelihood) for plotting
def read_log():
	file = open('log.txt', 'r')
	data = []
	for line in file.readlines():
		data.append(np.array(line.split(), dtype=float)[1:])
	file.close()
	return data

data = np.array(read_log()).T.reshape(NUMPES, NUMPES, 2, LEN_T)
# plot MSE
for i in range(NUMPES):
	for j in range(NUMPES):
		if i == j:
			plt.plot(t, np.log10(data[i][j][0]), label=r'$\rho_{%d%d}$' % (i, j))
		elif i < j:
			plt.plot(t, np.log10(data[i][j][0]), label=r'$\Re(\rho_{%d%d})$' % (i, j))
		else:
			plt.plot(t, np.log10(data[i][j][0]), label=r'$\Im(\rho_{%d%d})$' % (i, j))
plt.xlim((t[0],t[LEN_T-1]))
plt.title('log10 Mean Square Error')
plt.xlabel('t/a.u.')
plt.ylabel('lg(MSE)')
plt.legend(loc = 'best')
plt.savefig('mse.png')
plt.clf()
# plt -ln(likelihood)
for i in range(NUMPES):
	for j in range(NUMPES):
		if i == j:
			plt.plot(t, data[i][j][1], label=r'$\rho_{%d%d}$' % (i, j))
		elif i < j:
			plt.plot(t, data[i][j][1], label=r'$\Re(\rho_{%d%d})$' % (i, j))
		else:
			plt.plot(t, data[i][j][1], label=r'$\Im(\rho_{%d%d})$' % (i, j))
plt.xlim((t[0],t[LEN_T-1]))
plt.title('Negative Log Marginal Likelihood')
plt.xlabel('t/a.u.')
plt.ylabel('-ln(likelihood)')
plt.legend(loc = 'best')
plt.savefig('marg_ll.png')
plt.clf()
# plot their relationship
xsctt, ysctt = [], []
for i in range(NUMPES):
	for j in range(NUMPES):
		xsctt.append(np.log10(data[i][j][0]))
		ysctt.append(data[i][j][1])
index = []
xsctt = np.array(xsctt).reshape(NUMPES*NUMPES*LEN_T)
ysctt = np.array(ysctt).reshape(NUMPES*NUMPES*LEN_T)
for i in range(xsctt.size):
	if np.isinf(xsctt[i]) or np.isnan(xsctt[i]) or np.isinf(ysctt[i]) or np.isnan(ysctt[i]):
		index.append(i)
xsctt = np.delete(xsctt, index)
ysctt = np.delete(ysctt, index)
slope, intercept, rvalue, pvalue, stderr = stats.linregress(xsctt, ysctt)
xlin = np.array([xsctt.min(), xsctt.max()])
ylin = slope * xlin + intercept
fig, ax = plt.subplots()
ax.scatter(xsctt, ysctt, s=1, c='blue')
ax.plot(xlin, ylin, color='red')
if intercept > 0:
	ax.text(0.05, 0.95, "$y = %fx + %f$\n$R^2 = %f$" % (slope, intercept, rvalue*rvalue), va = 'top', ha = 'left', transform=ax.transAxes)
else:
	ax.text(0.05, 0.95, "$y = %fx - %f$\n$R^2 = %f$" % (slope, -intercept, rvalue*rvalue), va = 'top', ha = 'left', transform=ax.transAxes)
ax.set_xlabel('lg(MSE)')
ax.set_ylabel('-ln(likelihood)')
fig.savefig('relation.png')
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

# initialize the plot
fig, axs = plt.subplots(nrows=NUMPES, ncols=NUMPES*2, figsize=(40,10))

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
			axs[i][j].set_xlabel('x')
			axs[i][j].set_ylabel('p')
			if i == j // 2:
				if j % 2 == 0:
					axs[i][j].set_title(r'Actual $\rho_{%d%d}$' % (i, j // 2))
				else:
					axs[i][j].set_title(r'Simulated $\rho_{%d%d}$' % (i, j // 2))
			elif i < j // 2:
				if j % 2 == 0:
					axs[i][j].set_title(r'Actual $\Re(\rho_{%d%d})$' % (i, j // 2))
				else:
					axs[i][j].set_title(r'Simulated $\Re(\rho_{%d%d})$' % (i, j // 2))
			else:
				if j % 2 == 0:
					axs[i][j].set_title(r'Actual $\Im(\rho_{%d%d})$' % (j // 2, i))
				else:
					axs[i][j].set_title(r'Simulated $\Im(\rho_{%d%d})$' % (j // 2, i))
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
	for j in range(i - 1):
		for k in range(NUMPES*NUMPES):
			origin.readline() # actual rho
			sim.readline() # simulated rho
			choose.readline() # points
		origin.readline() # blank line
		sim.readline() # blank line
		choose.readline() # blank line
	# new data
	rho_orig = []
	rho_sim = []
	point = []
	for j in range(NUMPES):
		rho_orig.append([])
		rho_sim.append([])
		point.append([])
		for k in range(NUMPES):
			rho_orig[j].append(np.array(origin.readline().split(), dtype=float))
			rho_sim[j].append(np.array(sim.readline().split(), dtype=float))
			point[j].append(np.array(choose.readline().split(), dtype=float))
	origin.close()
	sim.close()
	choose.close()

	# adjust data to proper form
	LENGTH = len(rho_orig[0][0]) // 2
	NPOINT = len(point[0][0]) // 2
	for j in range(LENGTH):
		for k in range(NUMPES):
			rho_orig[k][k][j] = rho_orig[k][k][2*j]
			for l in range(k+1,NUMPES):
				rho_orig[k][l][j] = (rho_orig[k][l][2*j]+rho_orig[l][k][2*j])/2.0
				rho_orig[l][k][j] = (rho_orig[k][l][2*j+1]-rho_orig[l][k][2*j+1])/2.0
	for j in range(NUMPES):
		for k in range(NUMPES):
			rho_orig[j][k] = rho_orig[j][k][:LENGTH].reshape(LEN_P,LEN_X).T
			rho_sim[j][k] = rho_sim[j][k].reshape(LEN_P,LEN_X).T
			point[j][k] = point[j][k].reshape(NPOINT,2).T

	# print contourfs
	for j in range(NUMPES):
		for k in range(NUMPES):
			axs[j][2*k].contourf(xv, pv, rho_orig[j][k], levels=levels[j][k], cmap=cmap)
			axs[j][2*k].scatter(point[j][k][0], point[j][k][1], s=1, c='black')
			axs[j][2*k+1].contourf(xv, pv, rho_sim[j][k], levels=levels[j][k], cmap=cmap)
			axs[j][2*k+1].scatter(point[j][k][0], point[j][k][1], s=1, c='black')
	fig.suptitle(time_template % t[i])
	return fig, axs,

# make the animation
ani = animation.FuncAnimation(fig, ani, LEN_T, init, interval=10000//(t[1]-t[0]), repeat=False, blit=False)
# show
ani.save('phase.gif','imagemagick')
# plt.show()
