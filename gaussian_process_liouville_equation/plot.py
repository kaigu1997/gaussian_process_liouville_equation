import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import typing


HBAR = 1 # hbar in atomic unit
NUM_PES = 2 # number of potential energy surfaces
NUM_ELM = NUM_PES ** 2 # number of elements in density matrix
FIGSIZE = 5.0
CMAP = plt.get_cmap('seismic') # the kind of color: red-white-blue


def read_input(input_file: str) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def get_label(index: int) -> str:
	row, col = index // NUM_PES, index % NUM_PES
	if row == col:
		return r'$\rho_{%d,%d}$' % (row, col)
	elif row < col:
		return r'$\Re(\rho_{%d,%d})$' % (row, col)
	else:
		return r'$\Im(\rho_{%d,%d})$' % (col, row)


def plot_log(log_file: str, pic_file: str) -> np.ndarray:
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
	axs[0][0].set_ylabel('Error')
	axs[0][0].tick_params(axis='both')
	axs[0][0].legend(loc='best')
	axs[0][0].set_title('Error from Loose Function')
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
			label = get_label(i)
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


def plot_average(DIM: int, t: np.ndarray, ave_file: str, pic_file: str) -> None:
	X_LABEL = ['Analytical Integral', 'Direct Averaging', 'Monte Carlo Integral']
	Y_LABEL = [r'$x_{%d}$' % i for i in range(1, DIM + 1)] + [r'$p_{%d}$' % i for i in range(1, DIM + 1)] + ['Population', 'Energy', 'Purity']
	NUM_ROW, NUM_COL = len(Y_LABEL), len(X_LABEL) # for plot
	# get data
	data = np.loadtxt(ave_file)
	purity_total_data = data[:, -2:].reshape((t.size, 2))
	purity_elementwise_data = data[:, -(2 * NUM_ELM + 2):-2].reshape((t.size, 2, NUM_PES, NUM_PES))
	no_purity_data = data[:, 1:-(2 * NUM_ELM + 2)].reshape((t.size, NUM_PES + 1, NUM_COL, NUM_ROW - 1)) # apart from purity
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
		if X_LABEL[i] != 'Direct Averaging':
			DATA_INDEX = i // 2
			for j in range(NUM_PES):
				for k in range(NUM_PES):
					ax.plot(t, purity_elementwise_data[:, DATA_INDEX, j, k], label=r'$\rho_{%d,%d}$' % (j, k))
			ax.plot(t, purity_total_data[:, DATA_INDEX], label='Total')
			ax.set_xlabel('Time')
			ax.set_ylabel(Y_LABEL[NUM_ROW - 1])
			ax.set_title(X_LABEL[i] + ' of ' + Y_LABEL[NUM_ROW - 1])
			ax.legend(loc='best')
		else:
			ax.set_visible(False)
	# add title, save as png
	fig.suptitle('Averages')
	fig.savefig(pic_file)


def plot_param(DIM: int, t: np.ndarray, param_file: str, pic_file: str) -> None:
	Y_LABEL = ['Magnitude'] + [r'$x_{%d}$' % i for i in range(1, DIM + 1)] + [r'$p_{%d}$' % i for i in range(1, DIM + 1)] + ['Noise']
	CURVE_LABEL = ['Lower Bound', 'Data', 'Upper Bound']
	# get data
	data = np.loadtxt(param_file)
	NUM_VAR = data.shape[1]
	NUM_ROW, NUM_COL = NUM_VAR, NUM_ELM + 1 # for plot
	data = data.reshape((t.size, NUM_ELM, 3, NUM_VAR))
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE, NUM_ROW * FIGSIZE))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	# plot one by one
	for i in range(NUM_VAR):
		for j in range(NUM_ELM):
			ax = axs[i][j]
			if Y_LABEL[i] == 'Noise' or Y_LABEL[i] == 'Magnitude': # log scale for magnitude
				for k in range(3):
					ax.semilogy(t, data[:, j, k, i], label=CURVE_LABEL[k])
			else:
				for k in range(3):
					ax.plot(t, data[:, j, k, i], label=CURVE_LABEL[k])
			ax.set_xlabel('Time')
			ax.set_ylabel(Y_LABEL[i])
			ax.set_title(Y_LABEL[i] + ' of ' + get_label(j))
			ax.legend(loc='best')
		ax = axs[i][NUM_ELM]
		if Y_LABEL[i] == 'Noise' or Y_LABEL[i] == 'Magnitude': # log scale for magnitude
			for j in range(NUM_ELM):
				ax.semilogy(t, data[:, j, 1, i], label=get_label(j))
		else:
			for j in range(NUM_ELM):
				ax.plot(t, data[:, j, 1, i], label=get_label(j))
		ax.set_xlabel('Time')
		ax.set_ylabel(Y_LABEL[i])
		ax.set_title(Y_LABEL[i])
		ax.legend(loc='best')
	# add title, save as png
	fig.suptitle('Parameters')
	fig.savefig(pic_file)


def draw_point_anime(
	DIM: int,
	x: np.ndarray,
	p: np.ndarray,
	t: np.ndarray,
	parameter_point_file: str,
	extra_point_file: str,
	mcint_point_file: str,
	title: str,
	anime_both_file: str) -> np.ndarray:
	# general info and open files
	PHASEDIM = 2 * DIM
	prm_point = np.loadtxt(parameter_point_file).reshape((t.size, NUM_ELM, PHASEDIM, -1)) # prm = parameter
	xtr_point = np.loadtxt(extra_point_file).reshape((t.size, NUM_ELM, PHASEDIM, -1)) # xtr = extra
	mci_point = np.loadtxt(mcint_point_file).reshape((t.size, NUM_ELM, PHASEDIM, -1)) # mci = monte carlo integral
	xmin = min(x[0], prm_point[:, :, 0, :].min(), mci_point[:, :, 0, :].min())
	xmax = max(x[-1], prm_point[:, :, 0, :].max(), mci_point[:, :, 0, :].max())
	pmin = min(p[0], prm_point[:, :, 1, :].min(), mci_point[:, :, 1, :].min())
	pmax = max(p[-1], prm_point[:, :, 1, :].max(), mci_point[:, :, 1, :].max())
	xv, pv = np.meshgrid(np.linspace(xmin, xmax, 2), np.linspace(pmin, pmax, 2))
	# prepare for plot
	NUM_ROW, NUM_COL = NUM_PES, NUM_PES
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE, NUM_ROW * FIGSIZE))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	time_template = 'Time = %da.u.'

	# initialization: blank page
	def ani_init():
		# clear, then set x/y label, title of subplot, and colorbar
		for i in range(NUM_ELM):
			row, col = i // NUM_PES, i % NUM_PES
			axs[row][col].clear()
			axs[row][col].set_xlabel('x')
			axs[row][col].set_ylabel('p')
			axs[row][col].set_title(get_label(i))
			axs[row][col].contourf(xv, pv, np.zeros((2, 2)), colors='white')
		# figure settings: make them closer, title to be time
		fig.suptitle(title)
		return fig, axs,

	# plot frame by frame, check if parameter points scattered
	def ani_run(frame):
		if frame != 0:
			for i in range(NUM_ELM):
				row, col = i // NUM_PES, i % NUM_PES
				axs[row][col].contourf(xv, pv, np.zeros((2, 2)), colors='white')
				axs[row][col].scatter(mci_point[frame - 1, i, 0, :], mci_point[frame - 1, i, 1, :], s=3, c='red') # plot mci points
				axs[row][col].scatter(xtr_point[frame - 1, i, 0, :], xtr_point[frame - 1, i, 1, :], s=3, c='green') # plot xtr points
				axs[row][col].scatter(prm_point[frame - 1, i, 0, :], prm_point[frame - 1, i, 1, :], s=3, c='blue') # plot prm points
			fig.suptitle(title + '\n' + time_template % t[frame - 1])
		return fig, axs,

	mpl.animation.FuncAnimation(fig, ani_run, t.size, ani_init, interval=10, repeat=False, blit=False).save(anime_both_file, 'imagemagick')
	fig.clear()
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	# after animation, move out-of-range point into the box
	prm_point[:, :, 0, :] = np.where(prm_point[:, :, 0, :] < x[0], x[0], prm_point[:, :, 0, :])
	prm_point[:, :, 0, :] = np.where(prm_point[:, :, 0, :] > x[-1], x[-1], prm_point[:, :, 0, :])
	prm_point[:, :, 1, :] = np.where(prm_point[:, :, 1, :] < p[0], p[0], prm_point[:, :, 1, :])
	prm_point[:, :, 1, :] = np.where(prm_point[:, :, 1, :] > p[-1], p[-1], prm_point[:, :, 1, :])
	xtr_point[:, :, 0, :] = np.where(xtr_point[:, :, 0, :] < x[0], x[0], xtr_point[:, :, 0, :])
	xtr_point[:, :, 0, :] = np.where(xtr_point[:, :, 0, :] > x[-1], x[-1], xtr_point[:, :, 0, :])
	xtr_point[:, :, 1, :] = np.where(xtr_point[:, :, 1, :] < p[0], p[0], xtr_point[:, :, 1, :])
	xtr_point[:, :, 1, :] = np.where(xtr_point[:, :, 1, :] > p[-1], p[-1], xtr_point[:, :, 1, :])
	return np.append(prm_point, xtr_point, axis=3)


def draw_phase_anime(
	x: np.ndarray,
	p: np.ndarray,
	t: np.ndarray,
	point_file: np.ndarray,
	phase_file: typing.Union[str, np.ndarray],
	title: str,
	anime_point_file: str,
	anime_no_point_file: str,
	is_logscale: bool) -> np.ndarray:
	# general info
	NTICKS = 21
	LEN_X, LEN_P = x.size, p.size
	xv, pv = np.meshgrid(x, p)
	# prepare for plot
	NUM_ROW, NUM_COL = NUM_PES, NUM_PES
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE, NUM_ROW * FIGSIZE))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	time_template = 'Time = %da.u.'
	# open files
	point = point_file
	phase = (np.loadtxt(phase_file).reshape((t.size, NUM_ELM, LEN_X, LEN_P)) if type(phase_file) == str else phase_file)
	# construct the color bar
	if is_logscale:
		COLORBAR_LIMIT = np.max(np.abs(np.log(phase[phase > 0])))
		phase[phase <= 0] = np.exp(-COLORBAR_LIMIT)
		NORM = mpl.colors.LogNorm(np.exp(-COLORBAR_LIMIT), np.exp(COLORBAR_LIMIT), True)
		LEVEL = mpl.ticker.LogLocator(numdecs=1, numticks=NTICKS).tick_values(np.exp(-COLORBAR_LIMIT), np.exp(COLORBAR_LIMIT))
	else:
		COLORBAR_LIMIT = np.max(np.abs(phase))
		NORM = mpl.colors.CenteredNorm(0.0, COLORBAR_LIMIT, True)
		LEVEL = mpl.ticker.MaxNLocator(nbins=NTICKS).tick_values(-COLORBAR_LIMIT, COLORBAR_LIMIT)

	# initialization: blank page
	def ani_init():
		# clear, then set x/y label, title of subplot, and colorbar
		for i in range(NUM_ELM):
			row, col = i // NUM_PES, i % NUM_PES
			axs[row][col].clear()
			axs[row][col].set_xlabel('x')
			axs[row][col].set_ylabel('p')
			axs[row][col].set_title(get_label(i))
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
				axs[row][col].contourf(xv, pv, phase[frame - 1, i].T, levels=LEVEL, cmap=CMAP, norm=NORM) # plot contours
				if is_plot_point:
					axs[row][col].scatter(point[frame - 1, i, 0, :], point[frame - 1, i, 1, :], s=3, c='black') # plot points
			fig.suptitle(title + '\n' + time_template % t[frame - 1])
		return fig, axs,

	mpl.animation.FuncAnimation(fig, ani_run, t.size, ani_init, fargs=(True,), interval=200, repeat=False, blit=False).save(anime_point_file, 'imagemagick')
	fig.clear()
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	mpl.animation.FuncAnimation(fig, ani_run, t.size, ani_init, fargs=(False,), interval=200, repeat=False, blit=False).save(anime_no_point_file, 'imagemagick')
	return phase


if __name__ == '__main__':
	mass, xmin, xmax, pmin, pmax = read_input('input')
	DIM = mass.size
	LOG_FILE, LOG_PIC = 'run.log', 'log.png'
	AVE_FILE, AVE_PIC = 'ave.txt', 'ave.png'
	PRM_FILE, PRM_PIC = 'param.txt', 'param.png'
	PRM_PT_FILE, MCI_PT_FILE, XTR_PT_FILE = 'prm_point.txt', 'mci_point.txt', 'xtr_point.txt'
	PT_TITLE, ALL_PT_ANI = 'Points for Monte Carlo Integral', 'point.gif'
	PHS_FILE, PHS_TITLE, PHS_PT_ANI, PHS_NPT_ANI = 'phase.txt', 'Predicted Phase Space Distribution', 'phase_point.gif', 'phase_no_point.gif'
	VAR_FILE, VAR_TITLE, VAR_PT_ANI, VAR_NPT_ANI = 'var.txt', 'Prediction Variance', 'variance_point.gif', 'variance_no_point.gif'
	PHS_VAR_TITLE, PHS_VAR_PT_ANI, PHS_VAR_NPT_ANI = 'Predicted Distribution', 'phase_var_point.gif', 'phase_var_no_point.gif'

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
		pt_data = draw_point_anime(DIM, x, p, t, PRM_PT_FILE, XTR_PT_FILE, MCI_PT_FILE, PT_TITLE, ALL_PT_ANI)
		phs_data = draw_phase_anime(x, p, t, pt_data, PHS_FILE, PHS_TITLE, PHS_PT_ANI, PHS_NPT_ANI, False)
		var_data = draw_phase_anime(x, p, t, pt_data, VAR_FILE, VAR_TITLE, VAR_PT_ANI, VAR_NPT_ANI, True)
		phs_data[np.power(phs_data, 2) < var_data] = 0.0
		draw_phase_anime(x, p, t, pt_data, phs_data, PHS_VAR_TITLE, PHS_VAR_PT_ANI, PHS_VAR_NPT_ANI, False)
