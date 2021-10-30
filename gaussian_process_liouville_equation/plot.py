from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np


HBAR = 1 # hbar in atomic unit
NUM_PES = 2 # number of potential energy surfaces
NUM_ELM = NUM_PES ** 2 # number of elements in density matrix
FIGSIZE = 5
CMAP = plt.get_cmap('seismic') # the kind of color: red-white-blue
CLR_LIM = [0.4, 0.2, 0.2, 0.1]
LEVEL = [ticker.MaxNLocator(nbins=15).tick_values(-cl, cl) for cl in CLR_LIM]


def read_input(input_file):
	with open(input_file, 'r') as infile:
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
	dx = np.pi * HBAR / 2.0 / (p0 + 3.0 * sigma_p0)
	xNumGrids = ((xmax - xmin) / dx).astype(int)
	dx = (xmax - xmin) / xNumGrids.astype(int)
	pmax = p0 + np.pi * HBAR / 2.0 / dx
	pmin = p0 - np.pi * HBAR / 2.0 / dx
	return mass, xmin, xmax, pmin, pmax


def plot_log(log_file, pic_file):
	NUM_VAR = 4 + (NUM_ELM + 2) + 1 # error, population, autocor step and displacement, N**2+2 optimization steps, and time cost
	NUM_PLOT = 4 # error + population, autocor, steps, time cost
	NUM_COL = 2 # number of columns of axis
	NUM_ROW = NUM_PLOT // NUM_COL # number of rows of axis
	# get data
	data = np.loadtxt(log_file, usecols=np.linspace(0, NUM_VAR, NUM_VAR + 1, dtype=int))
	t, err, ppl, autocor_step, autocor_displace, steps, cputime = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], np.transpose(data[:, 5:-1]), data[:, -1]
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE, NUM_ROW * FIGSIZE))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	ax1_twin = axs[0][0].twinx()
	ax2_twin = axs[0][1].twinx()
	# plot error and population
	p1, = axs[0][0].semilogy(t, err, label='Error')
	ax1_twin.plot([], [])
	p2, = ax1_twin.plot(t, ppl, label='Normal Factor')
	axs[0][0].set_xlabel('Time')
	axs[0][0].set_ylabel('log(error)', color=p1.get_color())
	ax1_twin.set_ylabel('Norm', color=p2.get_color())
	axs[0][0].tick_params(axis='x')
	axs[0][0].tick_params(axis='y', colors=p1.get_color())
	ax1_twin.tick_params(axis='y', colors=p2.get_color())
	axs[0][0].legend(handles=[p1, p2], loc='best')
	axs[0][0].set_title('Error and Normalization')
	# plot autocorrelation
	p3, = axs[0][1].plot(t, autocor_step, label='Autocorrelation Steps')
	ax2_twin.plot([], [])
	p4, = ax2_twin.semilogy(t, autocor_displace, label='Autocorrelation Displacement')
	axs[0][1].set_xlabel('Time')
	axs[0][1].set_ylabel('Step', color=p3.get_color())
	ax2_twin.set_ylabel('log(Displacement)', color=p4.get_color())
	axs[0][1].tick_params(axis='x')
	axs[0][1].tick_params(axis='y', colors=p3.get_color())
	ax2_twin.tick_params(axis='y', colors=p4.get_color())
	axs[0][1].legend(handles=[p3, p4], loc='best')
	axs[0][1].set_title('Autocorrelation')
	# plot number of optimization steps
	for i in range(NUM_ELM + 2):
		if i < NUM_ELM:
			row, col = i // NUM_PES, i % NUM_PES
			if row == col:
				label = r'$\rho_{%d%d}$' % (row, col)
			elif row < col:
				label = r'$\Re(\rho_{%d%d})$' % (row, col)
			else:
				label = r'$\Im(\rho_{%d%d})$' % (col, row)
		elif i == NUM_ELM:
			label = 'Diagonal'
		else:
			label = 'Off-diagonal'
		axs[1][0].plot(t, steps[i], label=label)
	axs[1][0].set_xlabel('Time')
	axs[1][0].set_ylabel('Step')
	axs[1][0].tick_params(axis='both')
	axs[1][0].legend(loc='best')
	axs[1][0].set_title('Optimization Steps')
	# plot cpu time between each output
	axs[1][1].plot(t, cputime, label='CPU Time')
	axs[1][1].set_xlabel('Time/a.u.')
	axs[1][1].set_ylabel('CPU Time/s')
	axs[1][1].tick_params(axis='both')
	axs[1][1].legend(loc='best')
	axs[1][1].set_title('CPU Time between Outputs')
	# set title
	plt.suptitle('Evolution Log')
	# save file
	plt.savefig(pic_file)
	return t


def plot_average(DIM, t, ave_file, ave_pic, diff_pic):
	PHASEDIM = DIM * 2
	Y_LABEL = [['Population', 'Energy', 'Purity'], ['T', 'V', 'E']]
	NUM_ROW, NUM_COL = 1 + PHASEDIM + 1, 3 # for plot
	# get data
	data = np.loadtxt(ave_file)
	purity = data[:, -1]
	NUM_VAR = (data.shape[-1] - 2) // (NUM_PES + 1)
	data = data[:, 1:-1].reshape((-1, NUM_PES + 1, NUM_VAR)) # <1>, GPR <x> and <p>, MC <x> and <p>, <T>, <V>, <E>
	# prepare for average plot
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE, NUM_ROW * FIGSIZE))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	# plot first row: <1>, <E>, and purity
	INDICES = [0, -1]
	for i in range(NUM_COL):
		ax = axs[0][i]
		if i < len(INDICES):
			for j in range(NUM_PES):
				ax.plot(t, data[:, j, INDICES[i]], label='State %d' % j)
			ax.plot(t, data[:, NUM_PES, INDICES[i]], label='Total')
		else:
			ax.plot(t, purity, label='Purity')
		ax.set_xlabel('Time')
		ax.set_ylabel(Y_LABEL[0][i])
		ax.legend(loc='best')
	# plot second to second last row: x and p, compare GPR vs MC
	for i in range(PHASEDIM):
		if i < DIM:
			label = r'$x_{%d}$' % i
		else:
			label = r'$p_{%d}$' % (i % DIM)
		for j in range(2):
			if j == 0:
				ylabel = 'GPR ' + label
			else:
				ylabel = 'MC ' + label
			ax = axs[i + 1][j]
			for k in range(NUM_PES):
				ax.plot(t, data[:, k, j * PHASEDIM + 1 + i], label='State %d' % k)
			ax.plot(t, data[:, PHASEDIM, j * PHASEDIM + 1 + i], label='Total')
			ax.set_xlabel('Time')
			ax.set_ylabel(ylabel)
			ax.legend(loc='best')
		axs[i + 1][-1].set_visible(False)
	# plot last: <T> <V> <E>
	for i in range(NUM_COL):
		ax = axs[-1][i]
		for j in range(NUM_PES):
			ax.plot(t, data[:, j, i - 3], label='State %d' % j)
		ax.plot(t, data[:, NUM_PES, i - 3], label='Total')
		ax.set_xlabel('Time')
		ax.set_ylabel(Y_LABEL[-1][i])
		ax.legend(loc='best')
	# save as png
	fig.savefig(ave_pic)
	# prepare for difference plot
	plt.clf()
	NUM_ROW, NUM_COL = PHASEDIM, NUM_PES + 1
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE, NUM_ROW * FIGSIZE))
	axs = fig.subplots(nrows=PHASEDIM, ncols=NUM_PES + 1)
	# plot one by one
	for i in range(PHASEDIM):
		for j in range(NUM_PES + 1):
			ax = axs[i][j]
			ax_twin = ax.twinx()
			p1, = ax.plot(t, np.abs(data[:, j, i + 1] - data[:, j, i + 1 + PHASEDIM]))
			ax_twin.plot([], [])
			p2, = ax_twin.plot(t, np.abs(data[:, j, i + 1] / data[:, j, i + 1 + PHASEDIM] - 1.0))
			ax.set_xlabel('Time')
			ax.set_ylabel('Absolute Difference')
			ax_twin.set_ylabel('Relative Difference')
			ax.tick_params(axis='y', colors=p1.get_color())
			ax_twin.tick_params(axis='y', colors=p2.get_color())
			if i < DIM:
				str = r'$x_{%d}$' % i
			else:
				str = r'$p_{%d}$' % (i % DIM)
			str = 'Difference of ' + str
			if (j <= NUM_PES):
				str += ' of State %d' % j
			else:
				str += ' of Total'
			ax.set_title(str)
	fig.savefig(diff_pic)


def plot_param(DIM, t, param_file, pic_file):
	NUM_VAR = 2 + DIM * 2 # noise, magnitude, characteristic lengths
	Y_LABEL = ['Noise', 'log2(Magnitude)']
	for i in range(DIM):
		Y_LABEL.append(r'$x_{%d}$' % i)
	for i in range(DIM):
		Y_LABEL.append(r'$p_{%d}$' % i)
	NUM_ROW, NUM_COL = 1 + DIM, 2 # for plot
	# get data
	data = np.loadtxt(param_file).reshape((-1, NUM_ELM, NUM_VAR))
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE, NUM_ROW * FIGSIZE))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	# plot one by one
	for i in range(NUM_VAR):
		ax = axs[i // NUM_COL][i % NUM_COL]
		for j in range(NUM_ELM):
			ax.plot(t, data[:, j, i], label='State %d' % j)
		ax.set_xlabel('Time')
		ax.set_ylabel(Y_LABEL[i])
		ax.legend(loc='best')
	fig.savefig(pic_file)


def draw_anime(DIM, x, p, t, phase_file, point_file, anime_point_file, anime_no_point_file):
	# general info
	PHASEDIM = 2 * DIM
	LEN_X, LEN_P = x.size, p.size
	xv, pv = np.meshgrid(x, p)
	# prepare for plot
	NUM_ROW, NUM_COL = NUM_PES, NUM_PES
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE, NUM_ROW * FIGSIZE))
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

	# plot frame by frame, with points scattered
	def ani_with_points(frame):
		if frame != 0:
			for i in range(NUM_ELM):
				row, col = i // NUM_PES, i % NUM_PES
				place = NUM_ELM * (frame - 1) + i
				axs[row][col].contourf(xv, pv, phase[place].reshape(LEN_X, LEN_P).T, levels=LEVEL[i], cmap=CMAP) # plot contours
				axs[row][col].scatter(point[PHASEDIM * place], point[PHASEDIM * place + 1], s=3, c='black') # plot points
			fig.suptitle(time_template % t[frame - 1])
		return fig, axs,

	# plot frame by frame, without points scattered
	def ani_without_points(frame):
		if frame != 0:
			for i in range(NUM_ELM):
				row, col = i // NUM_PES, i % NUM_PES
				place = NUM_ELM * (frame - 1) + i
				axs[row][col].contourf(xv, pv, phase[place].reshape(LEN_X, LEN_P).T, levels=LEVEL[i], cmap=CMAP) # plot contours
			fig.suptitle(time_template % t[frame - 1])
		return fig, axs,

	anime = animation.FuncAnimation(fig, ani_with_points, t.size, ani_init, interval=10, repeat=False, blit=False)
	anime.save(anime_point_file, 'imagemagick')
	fig.clear()
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	anime = animation.FuncAnimation(fig, ani_without_points, t.size, ani_init, interval=10, repeat=False, blit=False)
	anime.save(anime_no_point_file, 'imagemagick')


if __name__ == '__main__':
	mass, xmin, xmax, pmin, pmax = read_input('input')
	DIM = mass.size
	LOG_FILE, LOG_PIC = 'run.log', 'log.png'
	AVE_FILE, AVE_PIC, DIFF_PIC = 'ave.txt', 'ave.png', 'diff.png'
	PRM_FILE, PRM_PIC = 'param.txt', 'param.png'
	PHS_FILE, PT_FILE, PHS_PT_PIC, PHS_NPT_PIC = 'phase.txt', 'point.txt', 'phase_point.gif', 'phase_no_point.gif'
	if DIM == 1: # plot
		# plot error and normalization factor
		t = plot_log(LOG_FILE, LOG_PIC)
		# plot averages
		plot_average(DIM, t, AVE_FILE, AVE_PIC, DIFF_PIC)
		# plot hyperparameters
		plot_param(DIM, t, PRM_FILE, PRM_PIC)
		# animation of evolution
		with open(PHS_FILE, 'r') as pf:
			NUM_GRID = int(np.sqrt(len(pf.readline().split())))
			pf.close()
		draw_anime(DIM, np.linspace(xmin, xmax, NUM_GRID), np.linspace(pmin, pmax, NUM_GRID), t, PHS_FILE, PT_FILE, PHS_PT_PIC, PHS_NPT_PIC)
