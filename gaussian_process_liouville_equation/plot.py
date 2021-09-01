from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np


NUM_PES = 2 # number of potential energy surfaces
NUM_ELM = NUM_PES ** 2 # number of elements in density matrix
FIGSIZE = 10
CMAP = plt.get_cmap('seismic') # the kind of color: red-white-blue
CLR_LIM = [0.4, 0.2, 0.2, 0.1]
LEVEL = [ticker.MaxNLocator(nbins=15).tick_values(-cl, cl) for cl in CLR_LIM]


def read_input(input_file):
	with open(input_file, "r") as infile:
		infile.readline()
		mass = np.array(infile.readline().split(), dtype=float)
		infile.readline()
		x0 = np.array(infile.readline().split(), dtype=float)
		infile.readline()
		p0 = np.array(infile.readline().split(), dtype=float)
		infile.readline()
		sigma_p0 = np.array(infile.readline().split(), dtype=float)
		infile.close()
	xmax = 2.0 * np.abs(x0)
	xmin = -xmax
	pmax = (p0 + sigma_p0) * 1.5
	pmin = (p0 - 3.0 * sigma_p0) * 0.5
	return mass, xmin, xmax, pmin, pmax


def plot_log(log_file, pic_file):
	NUM_VAR = 4 # error, population, autocor step, and autocor displacement
	NUM_LINE_PER_PLOT = 2 # number of lines on each plots (axis)
	NUM_PLOT = NUM_VAR // NUM_LINE_PER_PLOT # number of axis
	NUM_COL = 2 # number of columns of axis
	NUM_ROW = NUM_PLOT // NUM_COL # number of rows of axis
	# get data
	t, err, ppl, autocor_step, autocor_displace = np.loadtxt(log_file, usecols=np.linspace(0, NUM_VAR, NUM_VAR + 1, dtype=int), unpack=True)
	fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))
	ax1, ax2 = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	ax1_twin = ax1.twinx()
	ax2_twin = ax2.twinx()
	# plot
	p1, = ax1.semilogy(t, err, label="Error")
	ax1_twin.plot([], [])
	ax2.plot([], [])
	ax2_twin.plot([], [])
	p2, = ax1_twin.plot(t, ppl, label="Normal Factor")
	ax2.plot([], [])
	ax2_twin.plot([], [])
	p3, = ax2.plot(t, autocor_step, label="Autocorrelation Steps")
	ax2_twin.plot([], [])
	p4, = ax2_twin.plot(t, autocor_displace, label="Autocorrelation Displacement")
	# set label
	ax1.set_xlabel("Time")
	ax2.set_xlabel("Time")
	ax1.set_ylabel("log(error)", color=p1.get_color())
	ax1_twin.set_ylabel("Norm", color=p2.get_color())
	ax2.set_ylabel("Step", color=p3.get_color())
	ax2_twin.set_ylabel("Displacement", color=p4.get_color())
	# set tick
	ax1.tick_params(axis='x')
	ax2.tick_params(axis='x')
	ax1.tick_params(axis='y', colors=p1.get_color())
	ax1_twin.tick_params(axis='y', colors=p2.get_color())
	ax2.tick_params(axis='y', colors=p3.get_color())
	ax2_twin.tick_params(axis='y', colors=p4.get_color())
	# set legend
	ax1.legend(handles=[p1, p2], loc='best')
	ax2.legend(handles=[p3, p4], loc='best')
	# set title
	plt.suptitle("Evolution Log")
	# save file
	plt.savefig(pic_file)
	return t


def plot_average(t, ave_file, pic_file):
	NUM_VAR = 8 # ppl, x, p, T, V, E for each PES
	Y_LABEL = ["Population", "x", "p", "x", "p", "T", "V", "E"]
	NUM_ROW, NUM_COL = 2, 4 # for plot
	# get data
	data = np.loadtxt(ave_file, usecols=np.linspace(1, NUM_VAR * (NUM_PES + 1), NUM_VAR * (NUM_PES + 1), dtype=int)).reshape((-1, NUM_PES + 1, NUM_VAR))
	fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	# plot one by one
	for i in range(NUM_VAR):
		ax = axs[i // NUM_COL][i % NUM_COL]
		for j in range(NUM_PES):
			ax.plot(t, data[:, j, i], label="State %d" % j)
		ax.plot(t, data[:, NUM_PES, i], label="Total")
		ax.set_xlabel("Time")
		ax.set_ylabel(Y_LABEL[i])
		ax.legend(loc='best')
	fig.savefig(pic_file)


def plot_param(DIM, t, param_file, pic_file):
	NUM_VAR = 2 + 2 * DIM # noise, magnitude, characteristic lengths
	Y_LABEL = ["Noise", "Magnitude"]
	for i in range(DIM):
		Y_LABEL.append("x%d" % i)
		Y_LABEL.append("p%d" % i)
	NUM_ROW, NUM_COL = 1 + DIM, 2 # for plot
	# get data
	data = np.loadtxt(param_file).reshape((-1, NUM_ELM, NUM_VAR))
	fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	# plot one by one
	for i in range(NUM_VAR):
		ax = axs[i // NUM_COL][i % NUM_COL]
		for j in range(NUM_ELM):
			ax.plot(t, data[:, j, i], label="State %d" % j)
		ax.set_xlabel("Time")
		ax.set_ylabel(Y_LABEL[i])
		ax.legend(loc='best')
	fig.savefig(pic_file)


def draw_anime(DIM, x, p, t, phase_file, point_file, anime_file):
	# general info
	PHASEDIM = 2 * DIM
	LEN_X, LEN_P = x.size, p.size
	xv, pv = np.meshgrid(x, p)
	# prepare for plot
	NUM_ROW, NUM_COL = NUM_PES, NUM_PES
	fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	time_template = 'Time = %da.u.'
	# open files
	phase = np.loadtxt(phase_file)
	point = np.loadtxt(point_file)

	# initialization: blank page
	def ani_init():
		# clear, then set x/y label, title of subplot, and colorbar
		cfs = []
		for i in range(NUM_ELM):
			row, col = i // NUM_PES, i % NUM_PES
			axs[row][col].clear()
			axs[row][col].set_xlabel('x')
			axs[row][col].set_ylabel('p')
			if row == col:
				axs[row][col].set_title(r'$\rho_{%d%d}$' % (row, col))
			elif row < col:
				axs[row][col].set_title(r'$\Re(\rho_{%d%d})$' % (row, col))
			else:
				axs[row][col].set_title(r'$\Im(\rho_{%d%d})$' % (col, row))
			cfs.append(axs[row][col].contourf(xv, pv, np.zeros((LEN_X, LEN_P)), levels=LEVEL[i], cmap=CMAP))
			fig.colorbar(cfs[i], ax=axs[row][col])
			cfs[i].set_clim(-CLR_LIM[i], CLR_LIM[i])
		# figure settings: make them closer, title to be time
		fig.suptitle('')
		return fig, axs,

	# plot frame by frame
	def ani_frame(frame):
		if frame != 0:
			for i in range(NUM_ELM):
				row, col = i // NUM_PES, i % NUM_PES
				place = NUM_ELM * (frame - 1) + i
				axs[row][col].contourf(xv, pv, phase[place].reshape(LEN_X, LEN_P).T, levels=LEVEL[i], cmap=CMAP) # plot contours
				axs[row][col].scatter(point[PHASEDIM * place], point[PHASEDIM * place + 1], s=3, c='black') # plot points
			fig.suptitle(time_template % t[frame])
		return fig, axs,

	anime = animation.FuncAnimation(fig, ani_frame, t.size, ani_init, interval=10000 // (t[1] - t[0]), repeat=False, blit=False)
	anime.save(anime_file, 'imagemagick')


if __name__ == "__main__":
	mass, xmin, xmax, pmin, pmax = read_input("input")
	DIM = mass.size
	LOG_FILE, LOG_PIC = "run.log", "log.png"
	AVE_FILE, AVE_PIC = "ave.txt", "ave.png"
	PRM_FILE, PRM_PIC = "param.txt", "param.png"
	PHS_FILE, PT_FILE, PHS_PIC = "phase.txt", "point.txt", "phase.gif"
	if DIM == 1: # plot
		# plot error and normalization factor
		t = plot_log(LOG_FILE, LOG_PIC)
		# plot averages
		plot_average(t, AVE_FILE, AVE_PIC)
		# plot hyperparameters
		plot_param(DIM, t, PRM_FILE, PRM_PIC)
		# animation of evolution
		with open(PHS_FILE, "r") as pf:
			NUM_GRID = int(np.sqrt(len(pf.readline().split())))
			pf.close()
		draw_anime(DIM, np.linspace(xmin, xmax, NUM_GRID), np.linspace(pmin, pmax, NUM_GRID), t, PHS_FILE, PT_FILE, PHS_PIC)
