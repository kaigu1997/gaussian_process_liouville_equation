import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HBAR = 1 # hbar in atomic unit
NUM_PES = 2 # number of potential energy surfaces
NUM_ELM = NUM_PES ** 2 # number of elements in density matrix
FIGSIZE = 5
CMAP = plt.get_cmap('seismic') # the kind of color: red-white-blue


def read_input(input_file: str):
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


def plot_log(log_file: str, pic_file: str):
	NUM_VAR = 3 + (NUM_ELM + 2) + 1 # error, autocor step and displacement, N**2+2 optimization steps, and time cost
	# 4 plots: error, autocor, steps, time cost
	NUM_ROW = 2 # number of rows of axis
	NUM_COL = 2 # number of columns of axis
	# get data
	data = np.loadtxt(log_file, usecols=np.linspace(0, NUM_VAR, NUM_VAR + 1, dtype=int))
	t, err, autocor_step, autocor_displace, steps, cputime = data[:, 0], data[:, 1], data[:, 2], data[:, 3], np.transpose(data[:, 4:-1]), data[:, -1]
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE, NUM_ROW * FIGSIZE))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	# plot error
	axs[0][0].semilogy(t, err, label='Error')
	axs[0][0].set_xlabel('Time')
	axs[0][0].set_ylabel('log(error)')
	axs[0][0].tick_params(axis='x')
	axs[0][0].tick_params(axis='y')
	axs[0][0].legend(loc='best')
	axs[0][0].set_title('Error')
	# plot autocorrelation
	ax01_twin = axs[0][1].twinx()
	p1, = axs[0][1].plot(t, autocor_step, label='Autocorrelation Steps')
	ax01_twin.plot([], [])
	p2, = ax01_twin.semilogy(t, autocor_displace, label='Autocorrelation Displacement')
	axs[0][1].set_xlabel('Time')
	axs[0][1].set_ylabel('Step', color=p1.get_color())
	ax01_twin.set_ylabel('log(Displacement)', color=p2.get_color())
	axs[0][1].tick_params(axis='x')
	axs[0][1].tick_params(axis='y', colors=p1.get_color())
	ax01_twin.tick_params(axis='y', colors=p2.get_color())
	axs[0][1].legend(handles=[p1, p2], loc='best')
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
	axs[1][1].set_ylabel('Time/s')
	axs[1][1].tick_params(axis='both')
	axs[1][1].legend(loc='best')
	axs[1][1].set_title('CPU Time between Outputs')
	# set title
	fig.suptitle('Evolution Log')
	# save file
	plt.savefig(pic_file)
	return t


def plot_average(DIM: int, t, ave_file: str, pic_file: str) -> None:
	X_LABEL = ['Analytical Integral', 'Direct Averaging', 'Monte Carlo Integral']
	Y_LABEL = [r'$x_{%d}$' % i for i in range(1, DIM + 1)] + [r'$p_{%d}$' % i for i in range(1, DIM + 1)] + ['Population', 'Energy', 'Purity']
	NUM_ROW, NUM_COL = len(Y_LABEL), len(X_LABEL) # for plot
	# get data
	data = np.loadtxt(ave_file)
	purity_data = data[:, -3:].reshape((t.size, NUM_COL))
	no_purity_data = data[:, 1:-3].reshape((t.size, NUM_PES + 1, NUM_COL, NUM_ROW - 1)) # apart from purity
	# prepare for average plot
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE, NUM_ROW * FIGSIZE))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	# plot <r>, <1>, <E>
	for i in range(NUM_COL): # what kind of source: prm, ave, mci
		for j in range(NUM_ROW - 1): # which data: <r>, <1>, <E>
			ax = axs[j][i]
			if np.all(np.isfinite(no_purity_data[:, :, i, j])):
				for k in range(NUM_PES):
					ax.plot(t, no_purity_data[:, k, i, j], label='State %d' % k)
				ax.plot(t, no_purity_data[:, NUM_PES, i, j], label='Total')
				ax.set_xlabel('Time/a.u.')
				ax.set_ylabel(Y_LABEL[j] + '/a.u.')
				ax.set_title(X_LABEL[i] + ' of ' + Y_LABEL[j])
				ax.legend(loc='best')
			else:
				ax.set_visible(False)
	# plot purity
	for i in range(NUM_COL): # what kind of source: prm, ave, mci
		ax = axs[NUM_ROW - 1][i]
		if np.all(np.isfinite(purity_data[:, i])):
			ax.plot(t, purity_data[:, i], label='Total')
			ax.set_xlabel('Time')
			ax.set_ylabel(Y_LABEL[NUM_ROW - 1])
			ax.set_title(X_LABEL[i] + ' of ' + Y_LABEL[NUM_ROW - 1])
			ax.legend(loc='best')
		else:
			ax.set_visible(False)
	# add title, save as png
	fig.suptitle('Averages')
	fig.savefig(pic_file)


def plot_param(DIM: int, t, param_file: str, pic_file: str) -> None:
	NUM_VAR = 2 + DIM * 2 # noise, magnitude, characteristic lengths
	Y_LABEL = ['Noise', 'Magnitude'] + [r'$x_{%d}$' % i for i in range(1, DIM + 1)] + [r'$p_{%d}$' % i for i in range(1, DIM + 1)]
	NUM_ROW, NUM_COL = 1 + DIM, 2 # for plot
	# get data
	data = np.loadtxt(param_file).reshape((-1, NUM_ELM, NUM_VAR))
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE, NUM_ROW * FIGSIZE))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	# plot one by one
	for i in range(NUM_VAR):
		ax = axs[i // NUM_COL][i % NUM_COL]
		if Y_LABEL[i] == 'Magnitude': # log scale for magnitude
			for j in range(NUM_ELM):
				ax.semilogy(t, data[:, j, i], label='State %d' % j)
		else:
			for j in range(NUM_ELM):
				ax.plot(t, data[:, j, i], label='State %d' % j)
		ax.set_xlabel('Time')
		ax.set_ylabel(Y_LABEL[i])
		ax.legend(loc='best')
	# add title, save as png
	fig.suptitle('Parameters')
	fig.savefig(pic_file)


def draw_point_anime(DIM: int, x, p, t, parameter_point_file: str, mcint_point_file: str, title: str, anime_both_file: str, anime_mcint_file: str) -> None:
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
	prm_point = np.loadtxt(parameter_point_file) # prm = parameter
	mci_point = np.loadtxt(mcint_point_file) # mci = monte carlo integral

	# initialization: blank page
	def ani_init():
		# clear, then set x/y label, title of subplot, and colorbar
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
			axs[row][col].contourf(xv, pv, np.zeros((LEN_X, LEN_P)), colors='white')
		# figure settings: make them closer, title to be time
		fig.suptitle(title)
		return fig, axs,

	# plot frame by frame, check if parameter points scattered
	def ani_run(frame, is_plot_parameter_point):
		if frame != 0:
			for i in range(NUM_ELM):
				row, col = i // NUM_PES, i % NUM_PES
				place = NUM_ELM * (frame - 1) + i
				axs[row][col].contourf(xv, pv, np.zeros((LEN_X, LEN_P)), colors='white')
				axs[row][col].scatter(mci_point[PHASEDIM * place], mci_point[PHASEDIM * place + 1], s=3, c='red') # plot mci points
				if is_plot_parameter_point:
					axs[row][col].scatter(prm_point[PHASEDIM * place], prm_point[PHASEDIM * place + 1], s=3, c='blue') # plot prm points
			fig.suptitle(title + '\n' + time_template % t[frame - 1])
		return fig, axs,

	mpl.animation.FuncAnimation(fig, ani_run, t.size, ani_init, fargs=(True,), interval=10, repeat=False, blit=False).save(anime_both_file, 'imagemagick')
	fig.clear()
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	mpl.animation.FuncAnimation(fig, ani_run, t.size, ani_init, fargs=(False,), interval=10, repeat=False, blit=False).save(anime_mcint_file, 'imagemagick')


def draw_phase_anime(DIM: int, x, p, t, point_file: str, phase_file: str, title: str, anime_point_file: str, anime_no_point_file: str, is_logscale: bool) -> None:
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
	# construct the color bar
	if is_logscale:
		COLORBAR_LIMIT = np.max(np.abs(np.log(np.abs(phase[np.nonzero(phase)]))))
		phase[phase <= 0] = np.exp(-COLORBAR_LIMIT)
		NORM = mpl.colors.LogNorm(np.exp(-COLORBAR_LIMIT), np.exp(COLORBAR_LIMIT), True)
		LEVEL = mpl.ticker.LogLocator().tick_values(np.exp(-COLORBAR_LIMIT), np.exp(COLORBAR_LIMIT))
	else:
		COLORBAR_LIMIT = np.max(np.abs(phase))
		NORM = mpl.colors.CenteredNorm(0.0, COLORBAR_LIMIT, True)
		LEVEL = mpl.ticker.MaxNLocator(nbins=21).tick_values(-COLORBAR_LIMIT, COLORBAR_LIMIT)

	# initialization: blank page
	def ani_init():
		# clear, then set x/y label, title of subplot, and colorbar
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
			cfs = axs[row][col].contourf(xv, pv, np.ones((LEN_X, LEN_P)) if is_logscale else np.zeros((LEN_X, LEN_P)), levels=LEVEL, cmap=CMAP, norm=NORM)
			fig.colorbar(cfs, ax=axs[row][col], ticks=LEVEL)
		# figure settings: make them closer, title to be time
		fig.suptitle(title)
		return fig, axs,

	# plot frame by frame, check if points scattered
	def ani_run(frame, is_plot_point):
		if frame != 0:
			for i in range(NUM_ELM):
				row, col = i // NUM_PES, i % NUM_PES
				place = NUM_ELM * (frame - 1) + i
				axs[row][col].contourf(xv, pv, phase[place].reshape(LEN_X, LEN_P).T, levels=LEVEL, cmap=CMAP, norm=NORM) # plot contours
				if is_plot_point:
					axs[row][col].scatter(point[PHASEDIM * place], point[PHASEDIM * place + 1], s=3, c='black') # plot points
			fig.suptitle(title + '\n' + time_template % t[frame - 1])
		return fig, axs,

	mpl.animation.FuncAnimation(fig, ani_run, t.size, ani_init, fargs=(True,), interval=10, repeat=False, blit=False).save(anime_point_file, 'imagemagick')
	fig.clear()
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	mpl.animation.FuncAnimation(fig, ani_run, t.size, ani_init, fargs=(False,), interval=10, repeat=False, blit=False).save(anime_no_point_file, 'imagemagick')


if __name__ == '__main__':
	mass, xmin, xmax, pmin, pmax = read_input('input')
	DIM = mass.size
	LOG_FILE, LOG_PIC = 'run.log', 'log.png'
	AVE_FILE, AVE_PIC = 'ave.txt', 'ave.png'
	PRM_FILE, PRM_PIC = 'param.txt', 'param.png'
	PRM_PT_FILE, MC_PT_FILE, BOTH_PT_ANI, MC_PT_ANI = 'prm_point.txt', 'mci_point.txt', 'point.gif', 'mci_point.gif'
	PHS_FILE, PHS_PT_ANI, PHS_NPT_ANI = 'phase.txt', 'phase_point.gif', 'phase_no_point.gif'
	VAR_FILE, VAR_PT_ANI, VAR_NPT_ANI = 'var.txt', 'variance_point.gif', 'variance_no_point.gif'

	if DIM == 1: # plot
		# plot error and normalization factor
		t = plot_log(LOG_FILE, LOG_PIC)
		# plot averages
		plot_average(DIM, t, AVE_FILE, AVE_PIC)
		# plot hyperparameters
		plot_param(DIM, t, PRM_FILE, PRM_PIC)
		# animation of evolution
		with open(PHS_FILE, 'r') as pf:
			NUM_GRID = int(np.sqrt(len(pf.readline().split())))
			pf.close()
		x, p = np.linspace(xmin, xmax, NUM_GRID), np.linspace(pmin, pmax, NUM_GRID)
		draw_point_anime(DIM, x, p, t, PRM_PT_FILE, MC_PT_FILE, 'Points for Monte Carlo Integral', BOTH_PT_ANI, MC_PT_ANI)
		draw_phase_anime(DIM, x, p, t, PRM_PT_FILE, PHS_FILE, 'Predicted Phase Space Distribution', PHS_PT_ANI, PHS_NPT_ANI, False)
		draw_phase_anime(DIM, x, p, t, PRM_PT_FILE, VAR_FILE, 'Prediction Variance', VAR_PT_ANI, VAR_NPT_ANI, True)
