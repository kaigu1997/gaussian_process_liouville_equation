import importlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pkgutil
import types
import typing


def import_submodules(package: typing.Union[str, types.ModuleType], recursive: bool = True) -> typing.Dict[str, types.ModuleType]:
	"""Import all submodules of a module, recursively, including subpackages

	Args:
		package (typing.Union[str, types.ModuleType]): package (name or actual module)
		recursive (bool, optional): Whether to import packge recursively or not. Defaults to True.

	Returns:
		typing.Dict[str, types.ModuleType]: All submodules
	"""
	if isinstance(package, str):
		package = importlib.import_module(package)
	results: dict[str, types.ModuleType] = {}
	for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
		full_name: str = package.__name__ + '.' + name
		try:
			results[full_name] = importlib.import_module(full_name)
		except Exception:
			print(full_name)
		if recursive and is_pkg:
			results.update(import_submodules(full_name))
	return results


import_submodules(mpl, False)
HBAR: int = 1 # hbar in atomic unit
NUM_PES: int = 2 # number of potential energy surfaces
NUM_ELM: int = NUM_PES ** 2 # number of elements in density matrix
NUM_TRIG: int = (NUM_PES + NUM_ELM) // 2
FIGSIZE: tuple[float, float] = (6.4, 4.8)
CMAP = mpl.cm.get_cmap('seismic') # the kind of color: red-white-blue
NTICKS: int = 21


def read_input(input_file: str) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
	"""Read input information for GPR-MQCL

	Args:
		input_file (str): Name of input file

	Returns:
		typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: mass, min and max of position, min and max of momentum
	"""
	with open(input_file, 'r') as infile:
		infile.readline()
		mass = np.array(infile.readline().split(), dtype=float)
		infile.readline()
		x0 = np.array(infile.readline().split(), dtype=float)
		infile.readline()
		p0 = np.array(infile.readline().split(), dtype=float)
		infile.readline()
		sigma_p0 = np.array(infile.readline().split(), dtype=float)
		infile.readline()
		output_time = float(infile.readline())
		infile.readline()
		reopt_time = float(infile.readline())
		infile.readline()
		dt = float(infile.readline())
		infile.readline()
		num_points = int(infile.readline())
	xmax = 2.0 * np.abs(x0)
	xmin = -xmax
	dx = np.pi * HBAR / 2.0 / (p0 + 3.0 * sigma_p0)
	xNumGrids = ((xmax - xmin) / dx).astype(int)
	dx = (xmax - xmin) / xNumGrids.astype(int)
	pmax = p0 + np.pi * HBAR / 2.0 / dx
	pmin = p0 - np.pi * HBAR / 2.0 / dx
	return mass, xmin, xmax, pmin, pmax, num_points


def get_element_label(row: int, col: int) -> str:
	"""Return the label of the corresponding element

	Args:
		row (int): The row index of the element, range in [0, NUM_PES)
		col (int): The column index of the element, generally range in [0, row]

	Returns:
		str: The latex formatted name of the element
	"""
	return r'$\rho_{%d,%d}$' % (row, col)


def get_RI_label(row: int, col: int) -> str:
	"""Return the label of the corresponding real or imaginary part

	Args:
		row (int): The row index of the element, range in [0, NUM_PES)
		col (int): The column index of the element, range in [0, NUM_PES)

	Returns:
		str: The latex formatted name of the real or imaginary part of the element
	"""
	if row == col:
		return r'$\rho_{%d,%d}$' % (row, col)
	elif row < col:
		return r'$\Re\rho_{%d,%d}$' % (col, row)
	else:
		return r'$\Im\rho_{%d,%d}$' % (row, col)


def get_index(row: int, col: int) -> int:
	"""Return the index in the lower-triangular matrix

	Args:
		row (int): The row index of the element, range in [0, NUM_PES)
		col (int): The column index of the element, range in [0, row]

	Returns:
		int: The index in the array of linearized lower-triangular matrix
	"""
	return row * (row + 1) // 2 + col


def plot_log(log_file: str, pic_file: str) -> np.ndarray:
	"""To plot the log data, including total error, autocorrelation, steps for optimization and CPU time

	Args:
		log_file (str): input file name
		pic_file (str): output file name

	Returns:
		np.ndarray: all time points, for future plotting
	"""
	OPT_TYPE_LABEL: list[str] = ['Local with Previous', 'Local with Initial', 'Global']
	OPT_TYPE_VAL = np.arange(1, len(OPT_TYPE_LABEL) + 1).tolist()
	TITLES: list[list[str]] = [['CPU Time between Outputs', 'Metropolis Monte Carlo Parameters', 'Rescale Factor'], ['Error from Loose Function', 'Optimization Steps', 'Selected Optimization Type']]
	Y_LABEL: list[list[str]] = [['Time/s', '', 'Rescale Factor'], ['Error', 'Step', 'Optimization Type']]
	NUM_VAR: int = 2 + 3 * NUM_TRIG + 1 + (NUM_TRIG + 2) + 1 # time, time cost, autocor step and displacement, rescale factor, error, opt steps, opt type
	# 6 plots: error, autocor, steps, time cost
	NUM_ROW: int = 2 # number of rows of axis
	NUM_COL: int = 3 # number of columns of axis
	# get data
	data = np.loadtxt(log_file, usecols=np.arange(NUM_VAR))
	idx = 0
	t = data[:, idx]
	idx += 1
	cputime = data[:, idx]
	idx += 1
	mc_step = data[:, idx:idx + NUM_TRIG].T
	idx += NUM_TRIG
	mc_displace = data[:, idx:idx + NUM_TRIG].T
	idx += NUM_TRIG
	rescale_factor = data[:, idx:idx + NUM_TRIG].T
	idx += NUM_TRIG
	err = data[:, idx]
	idx += 1
	steps = data[:, idx:idx + NUM_TRIG + 2].T
	idx += NUM_TRIG + 2
	opt_type = data[:, idx]
	idx += 1
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE[0], NUM_ROW * FIGSIZE[1]))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL, squeeze=False)
	# plot cputime
	# plot cpu time between each output
	axs[0][0].semilogy(t, cputime, label='CPU Time')
	# plot autocorrelation
	mc_step_lines, mc_displace_lines = [], []
	ax01_twin = axs[0][1].twinx()
	for iPES in range(NUM_PES):
		for jPES in range(iPES + 1):
			mc_step_lines += axs[0][1].plot(t, mc_step[get_index(iPES, jPES)], label='Best number of steps of ' + get_element_label(iPES, jPES))
			ax01_twin.plot([], []) # blank plot for color delay
	for iPES in range(NUM_PES):
		for jPES in range(iPES + 1):
			mc_displace_lines += ax01_twin.semilogy(t, mc_displace[get_index(iPES, jPES)], label='Best displacement of ' + get_element_label(iPES, jPES))
	axs[0][1].set_ylabel('Step', color=mc_step_lines[0].get_color())
	ax01_twin.set_ylabel('Displacement', color=mc_displace_lines[0].get_color())
	axs[0][1].tick_params(axis='y', which='both', colors=mc_step_lines[0].get_color())
	ax01_twin.tick_params(axis='y', which='both', colors=mc_displace_lines[0].get_color())
	axs[0][1].legend(handles=mc_step_lines + mc_displace_lines, loc='best')
	# plot rescale factor
	for iPES in range(NUM_PES):
		for jPES in range(iPES + 1):
			axs[0][2].semilogy(t, rescale_factor[get_index(iPES, jPES)], label=get_element_label(iPES, jPES))
	# plot error
	axs[1][0].semilogy(t, err, label='Error')
	# plot number of optimization steps
	for iPES in range(NUM_PES):
		for jPES in range(iPES + 1):
			axs[1][1].plot(t, steps[get_index(iPES, jPES)], label=get_element_label(iPES, jPES))
	axs[1][1].plot(t, steps[NUM_TRIG], label='Diagonal')
	axs[1][1].plot(t, steps[NUM_TRIG + 1], label='Full')
	# plot optimization type whose result is selected
	axs[1][2].plot(t, opt_type, label='Optimization Type')
	axs[1][2].set_ylim(min(OPT_TYPE_VAL) - 1, max(OPT_TYPE_VAL) + 1)
	axs[1][2].set_yticks(OPT_TYPE_VAL)
	axs[1][2].set_yticklabels([label + ('\n \n ' if i % 2 == 0 else '') for i, label in enumerate(OPT_TYPE_LABEL)], horizontalalignment='center', multialignment='center', rotation='vertical', rotation_mode='anchor')
	# set tick, x axis and and legend
	for i in range(NUM_ROW):
		for j in range(NUM_COL):
			if i != 0 or j != 1:
				axs[i][j].legend(loc='best')
				axs[i][j].set_ylabel(Y_LABEL[i][j])
			axs[i][j].set_xlabel('Time/a.u.')
			axs[i][j].set_title(TITLES[i][j])
	# set title
	fig.suptitle('Evolution Log')
	fig.set_layout_engine('compressed')
	# save file
	plt.savefig(pic_file)
	return t


def plot_average(DIM: int, t: np.ndarray, ave_file: str, pic_file: str) -> None:
	"""To plot averages (population, <x> and <p>, energy, and purity) over time

	Args:
		DIM (int): Number of dimensions of classical degree of freedom
		t (np.ndarray): All time ticks
		ave_file (str): Input file name
		pic_file (str): Output file name
	"""
	X_LABEL: list[str] = ['Analytical Integral', 'Monte Carlo Integral']
	Y_LABEL: list[str] = ['Population'] + [r'$x_{%d}$' % i for i in range(1, DIM + 1)] + [r'$p_{%d}$' % i for i in range(1, DIM + 1)] + ['Energy']
	NUM_ROW, NUM_COL = len(Y_LABEL) + 1, len(X_LABEL) + 1 # for plot, +1 row for purity, +1 col for total (in case special value from some element)
	NUM_AVE = len(Y_LABEL) * len(X_LABEL) * (NUM_PES + 1)
	# get data
	data = np.loadtxt(ave_file)
	TOTAL_TIME = data.shape[0]
	ave_data = data[:, :NUM_AVE].reshape((TOTAL_TIME, NUM_PES + 1, len(X_LABEL), len(Y_LABEL)))
	purity_data = data[:, NUM_AVE:NUM_AVE + (NUM_ELM + 1) * len(X_LABEL)].reshape((TOTAL_TIME, len(X_LABEL), NUM_ELM + 1))
	# prepare for average plot
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE[0], NUM_ROW * FIGSIZE[1]))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL, squeeze=False)
	# plot <1>, <r>, <E>
	for j in range(len(Y_LABEL)): # which data: <1>, <r>, <E>
		for i in range(len(X_LABEL)): # what kind of source: prm, ave
			ax = axs[j][i]
			for k in range(NUM_PES):
				# no analytically integrated energy
				if not np.isnan(ave_data[:, k, i, j]).all():
					ax.plot(t[:TOTAL_TIME], ave_data[:, k, i, j], label='State %d' % k)
			ax.plot(t[:TOTAL_TIME], ave_data[:, NUM_PES, i, j], label='Total')
			ax.set_xlabel('Time/a.u.')
			ax.set_ylabel(Y_LABEL[j] + '/a.u.' if j != 0 else '')
			ax.set_title(X_LABEL[i] + ' of ' + Y_LABEL[j])
			ax.legend(loc='best')
			# plot for total
			axs[j][len(X_LABEL)].plot(t[:TOTAL_TIME], ave_data[:, NUM_PES, i, j], label=X_LABEL[i])
		axs[j][len(X_LABEL)].set_xlabel('Time/a.u.')
		axs[j][len(X_LABEL)].set_ylabel(Y_LABEL[j] + '/a.u.' if j != 0 else '')
		axs[j][len(X_LABEL)].set_title(Y_LABEL[j])
		axs[j][len(X_LABEL)].legend(loc='best')
	# plot purity
	for i in range(len(X_LABEL)): # what kind of source: prm, ave, mci
		ax = axs[len(Y_LABEL)][i]
		for j in range(NUM_PES):
			for k in range(NUM_PES):
				ax.plot(t[:TOTAL_TIME], purity_data[:, i, j * NUM_PES + k], label=get_element_label(j, k))
		ax.plot(t[:TOTAL_TIME], purity_data[:, i, NUM_ELM], label='Total')
		ax.set_xlabel('Time')
		ax.set_ylabel('Purity')
		ax.set_title(X_LABEL[i] + ' of Purity')
		ax.legend(loc='best')
		axs[len(Y_LABEL)][len(X_LABEL)].plot(t[:TOTAL_TIME], purity_data[:, i, NUM_ELM], label=X_LABEL[i])
	axs[len(Y_LABEL)][len(X_LABEL)].set_xlabel('Time/a.u.')
	axs[len(Y_LABEL)][len(X_LABEL)].set_ylabel('Purity')
	axs[len(Y_LABEL)][len(X_LABEL)].set_title('Purity')
	axs[len(Y_LABEL)][len(X_LABEL)].legend(loc='best')
	# add title, save as png
	fig.suptitle('Averages')
	fig.savefig(pic_file)


def plot_param(DIM: int, t: np.ndarray, param_file: str, pic_file: str) -> None:
	"""To plot parameters (population, <x> and <p>, energy, and purity) over time

	Args:
		DIM (int): Number of dimensions of classical degree of freedom
		t (np.ndarray): All time ticks
		ave_file (str): Input file name
		pic_file (str): Output file name
	"""
	COMMON_PART = ['Magnitude'] + [r'$x_{%d}$' % i for i in range(1, DIM + 1)] + [r'$p_{%d}$' % i for i in range(1, DIM + 1)]
	Y_LABEL_DIAG = COMMON_PART + ['Noise']
	Y_LABEL_OFFDIAG = ['Magnitude'] + ['Real ' + s for s in COMMON_PART] + ['Imaginary ' + s for s in COMMON_PART] + ['Noise']
	CURVE_LABEL = ['Lower Bound', 'Data', 'Upper Bound']
	# get data
	data = np.zeros((t.size * (NUM_TRIG * 3 + 1), max(len(Y_LABEL_OFFDIAG), len(Y_LABEL_DIAG))), dtype=float)
	with open(param_file, 'r') as f:
		for i, line in enumerate(f.readlines()):
			if line != '\n':
				s = line[:-1].split(' ')
				data[i][:len(s)] = np.asarray(s, dtype=float)
	data = np.delete(data, np.arange(NUM_TRIG * 3, data.shape[0], NUM_TRIG * 3 + 1), axis=0)
	NUM_VAR = data.shape[1]
	NUM_ROW, NUM_COL = NUM_VAR, NUM_TRIG # for plot, only lower-triangular part
	data = data.reshape((t.size, NUM_TRIG, 3, NUM_VAR))
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE[0], NUM_ROW * FIGSIZE[1]))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL, squeeze=False)
	# plot one by one
	for iPES in range(NUM_PES):
		for jPES in range(iPES + 1):
			idx = get_index(iPES, jPES)
			for iVar in range(NUM_VAR):
				ax = axs[iVar][idx]
				ELM_VAR: list[str] = Y_LABEL_DIAG if iPES == jPES else Y_LABEL_OFFDIAG
				if iVar < len(ELM_VAR):
					if 'Noise' in ELM_VAR[iVar] or 'Magnitude' in ELM_VAR[iVar]: # log scale for magnitude
						for iCurve in range(3):
							ax.semilogy(t, data[:, idx, iCurve, iVar], label=CURVE_LABEL[iCurve])
					else:
						for iCurve in range(3):
							ax.plot(t, data[:, idx, iCurve, iVar], label=CURVE_LABEL[iCurve])
					ax.set_xlabel('Time')
					ax.set_ylabel(ELM_VAR[iVar])
					ax.set_title(ELM_VAR[iVar] + ' of ' + get_element_label(iPES, jPES))
					ax.legend(loc='best')
				else:
					ax.set_visible(False)
	# add title, save as png
	fig.suptitle('Parameters')
	fig.savefig(pic_file)


def plot_point_error(t: np.ndarray, num_points: int, value_file: str) -> None:
	"""To plot the difference of density of seleted points between initial and given time

	Args:
		t (np.ndarray): All time ticks
		num_points (int): The first N points for gaussian process
		value_file (str): Input file name, containing density of all points
	"""
	value = np.loadtxt(value_file).reshape((t.size, NUM_TRIG, 2, -1))[..., :num_points]
	fig = plt.figure()
	ax = fig.subplots(nrows=1, ncols=1, squeeze=True)
	for iPES in range(NUM_PES):
		for jPES in range(iPES + 1):
			trig_index = get_index(iPES, jPES)
			data = np.copy(value[:, trig_index, 0, :]) if iPES == jPES else np.sqrt(np.square(value[:, trig_index, :, :]).sum(axis=1))
			data -= data[0]
			ax.plot(t, data.max(axis=1), label='Maximum Positive Error for ' + get_element_label(iPES, jPES))
			ax.plot(t, data.min(axis=1), label='Maximum Negative Error for ' + get_element_label(iPES, jPES))
	ax.set_xlabel('Time')
	ax.set_ylabel('Error')
	ax.set_title('Difference of Adiabatic Density Compared with Initial Density')
	fig.legend()
	fig.savefig('error.png')


def draw_point_anime(
	DIM: int,
	x: np.ndarray,
	p: np.ndarray,
	t: np.ndarray,
	num_points: int,
	point_file: str,
	value_file: str,
	density_title: str,
	density_file: str,
	extra_point_title: str,
	extra_point_file: str) -> np.ndarray:
	"""To draw the animation of all selected points

	Args:
		DIM (int): Number of dimensions of classical degree of freedom
		x (np.ndarray): Position ticks
		p (np.ndarray): Momentum ticks
		t (np.ndarray): All time ticks
		num_points (int): The first N points for gaussian process
		point_file (str): Input file name, containing coordinates of all points
		value_file (str): Input file name, containing density of all points
		density_title (str): The title of the figure of points for GPR
		density_file (str): Output file name of points for GPR
		extra_point_title (str): The title of the figure of points for optimization
		extra_point_file (str): Output file name of points for optimization

	Returns:
		np.ndarray: All selected points in the range of [x[0], x[-1]] and [p[0], p[-1]]
	"""
	# general info and open files
	PHASEDIM = 2 * DIM
	point = np.loadtxt(point_file).reshape((t.size, NUM_TRIG, PHASEDIM, -1))
	NUM_TOTAL_POINTS = point.shape[-1]
	value = np.loadtxt(value_file).reshape((t.size, NUM_TRIG, 2, -1))
	xmin = min(x[0], point[:, :, 0, :].min())
	xmax = max(x[-1], point[:, :, 0, :].max())
	pmin = min(p[0], point[:, :, 1, :].min())
	pmax = max(p[-1], point[:, :, 1, :].max())
	COLORBAR_LIMIT = np.max(np.abs(value))
	NORM = mpl.colors.CenteredNorm(0.0, COLORBAR_LIMIT, True)
	LEVEL = mpl.ticker.MaxNLocator(nbins=NTICKS).tick_values(-COLORBAR_LIMIT, COLORBAR_LIMIT)
	value /= COLORBAR_LIMIT
	xv, pv = np.meshgrid(np.linspace(xmin, xmax, 2), np.linspace(pmin, pmax, 2))
	# prepare for plot
	NUM_ROW, NUM_COL = NUM_PES, NUM_PES
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE[0], NUM_ROW * FIGSIZE[1]))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL, squeeze=False)
	time_template = 'Time = %fa.u.'

	def ani_init() -> typing.Iterable[mpl.artist.Artist]:
		"""Initialize the animation - setting labels, names, white background

		Returns:
			typing.Iterable[mpl.artist.Artist]: Figure and Axes, which are inherited from mpl.artist
		"""
		# clear, then set x/y label, title of subplot, and colorbar
		for iPES in range(NUM_PES):
			for jPES in range(NUM_PES):
				axs[iPES][jPES].clear()
				axs[iPES][jPES].set_xlabel('x')
				axs[iPES][jPES].set_ylabel('p')
				axs[iPES][jPES].set_title(get_RI_label(iPES, jPES))
				axs[iPES][jPES].contourf(xv, pv, np.zeros((2, 2)), colors='white')
		# figure settings: make them closer, title to be time
		fig.colorbar(mpl.cm.ScalarMappable(cmap=CMAP, norm=NORM), ax=axs.ravel().tolist(), ticks=LEVEL)
		return fig, axs,

	def ani_run(frame: int, is_training_set: bool) -> typing.Iterable[mpl.artist.Artist]:
		"""Plot frame by frame

		Args:
			frame (int): Using the corresponding data. Notice t begins from 0 but frame begins from 1
			is_training_set (bool): Whether the points are from training set or not

		Returns:
			typing.Iterable[mpl.artist.Artist]: Figure and Axes, which are inherited from mpl.artist
		"""
		if frame != 0:
			time = frame - 1
			for iPES in range(NUM_PES):
				for jPES in range(NUM_PES):
					idx = get_index(max(iPES, jPES), min(iPES, jPES))
					axs[iPES][jPES].contourf(xv, pv, np.zeros((2, 2)), colors='white')
					# calculate RGB color. Larger value <-> darker color
					selected_points = point[time, idx, :, 0 if is_training_set else num_points:num_points if is_training_set else NUM_TOTAL_POINTS]
					color = np.ones((selected_points.shape[-1], 3))
					elm_clr = value[time, idx, 0 if iPES <= jPES else 1, 0 if is_training_set else num_points:num_points if is_training_set else NUM_TOTAL_POINTS]
					color[:, 0] -= np.where(elm_clr < 0, -elm_clr, 0)
					color[:, 1] -= np.abs(elm_clr)
					color[:, 2] -= np.where(elm_clr > 0, elm_clr, 0)
					axs[iPES][jPES].scatter(selected_points[0, :], selected_points[1, :], s=3, c=color) # plot points
			fig.suptitle((density_title if is_training_set else extra_point_title) + '\n' + time_template % t[time])
		return fig, axs,

	mpl.animation.FuncAnimation(fig, ani_run, t.size, ani_init, fargs=(True,), interval=100, repeat=False, blit=False).save(density_file, 'imagemagick')
	fig.clear()
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	mpl.animation.FuncAnimation(fig, ani_run, t.size, ani_init, fargs=(False,), interval=100, repeat=False, blit=False).save(extra_point_file, 'imagemagick')
	# after animation, move out-of-range point into the box
	point[:, :, 0, :] = np.where(point[:, :, 0, :] < x[0], x[0], point[:, :, 0, :])
	point[:, :, 0, :] = np.where(point[:, :, 0, :] > x[-1], x[-1], point[:, :, 0, :])
	point[:, :, 1, :] = np.where(point[:, :, 1, :] < p[0], p[0], point[:, :, 1, :])
	point[:, :, 1, :] = np.where(point[:, :, 1, :] > p[-1], p[-1], point[:, :, 1, :])
	return point


def draw_phase_anime(
	x: np.ndarray,
	p: np.ndarray,
	t: np.ndarray,
	point: np.ndarray,
	phase: np.ndarray,
	title: str,
	anime_point_file: str,
	anime_no_point_file: str,
	is_logscale: bool) -> None:
	"""To draw animation of contours of phase space distribution

	Args:
		x (np.ndarray): Position ticks
		p (np.ndarray): Momentum ticks
		t (np.ndarray): All time ticks
		point (np.ndarray): All selected points
		phase_file (np.ndarray): All phase grids, or the file containing it
		title (str): The name of the figure
		anime_point_file (str): Output file name, which the selected points are NOT scattered
		anime_no_point_file (str): Output file name, which the selected points are scattered
		is_logscale (bool): Whether to plot in logscale (for variance) or not (for distribution)

	Returns:
		np.ndarray: The phase space gridded distribution, i.e. what is in phase_file
	"""
	# general info
	LEN_X, LEN_P = x.size, p.size
	xv, pv = np.meshgrid(x, p)
	# prepare for plot
	NUM_ROW, NUM_COL = NUM_PES, NUM_PES
	fig = plt.figure(figsize=(NUM_COL * FIGSIZE[0], NUM_ROW * FIGSIZE[1]))
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL, squeeze=False)
	time_template = 'Time = %fa.u.'
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

	def ani_init() -> typing.Iterable[mpl.artist.Artist]:
		"""Initialize the animation - setting labels, names, white background

		Returns:
			typing.Iterable[mpl.artist.Artist]: Figure and Axes, which are inherited from mpl.artist
		"""
		# clear, then set x/y label, title of subplot, and colorbar
		if is_logscale:
			for iPES in range(NUM_PES):
				for jPES in range(iPES + 1):
					axs[iPES][jPES].clear()
					axs[iPES][jPES].set_xlabel('x')
					axs[iPES][jPES].set_ylabel('p')
					axs[iPES][jPES].set_title(get_element_label(iPES, jPES))
					axs[iPES][jPES].contourf(xv, pv, np.ones((LEN_X, LEN_P)), levels=LEVEL, cmap=CMAP, norm=NORM)
				for jPES in range(iPES + 1, NUM_PES):
					axs[iPES][jPES].contourf(xv, pv, np.ones((LEN_X, LEN_P)), levels=LEVEL, cmap=CMAP, norm=NORM)
					axs[iPES][jPES].set_axis_off()
		else:
			for iPES in range(NUM_PES):
				for jPES in range(NUM_PES):
					axs[iPES][jPES].clear()
					axs[iPES][jPES].set_xlabel('x')
					axs[iPES][jPES].set_ylabel('p')
					axs[iPES][jPES].set_title(get_RI_label(iPES, jPES))
					axs[iPES][jPES].contourf(xv, pv, np.zeros((LEN_X, LEN_P)), levels=LEVEL, cmap=CMAP, norm=NORM)
		# figure settings: make them closer, title to be time
		fig.colorbar(mpl.cm.ScalarMappable(cmap=CMAP, norm=NORM), ax=axs.ravel().tolist(), ticks=LEVEL)
		fig.suptitle(title)
		return fig, axs,

	def ani_run(frame: int, is_plot_point: bool) -> typing.Iterable[mpl.artist.Artist]:
		"""Plot frame by frame

		Args:
			frame (int): Using the corresponding data. Notice t begins from 0 but frame begins from 1
			is_plot_point (bool): Whether to scatter the points or not

		Returns:
			typing.Iterable[mpl.artist.Artist]: Figure and Axes, which are inherited from mpl.artist
		"""
		if frame != 0:
			time = frame - 1
			if is_logscale:
				for iPES in range(NUM_PES):
					for jPES in range(iPES + 1):
						idx = get_index(iPES, jPES)
						axs[iPES][jPES].contourf(xv, pv, phase[time, idx].T, levels=LEVEL, cmap=CMAP, norm=NORM) # plot contours
						if is_plot_point:
							axs[iPES][jPES].scatter(point[time, idx, 0, :], point[time, idx, 1, :], s=2, c='black') # plot points
			else:
				for iPES in range(NUM_PES):
					for jPES in range(NUM_PES):
						idx = get_index(max(iPES, jPES), min(iPES, jPES))
						axs[iPES][jPES].contourf(xv, pv, phase[time, idx, 0 if iPES <= jPES else 1].T, levels=LEVEL, cmap=CMAP, norm=NORM) # plot contours
						if is_plot_point:
							axs[iPES][jPES].scatter(point[time, idx, 0, :], point[time, idx, 1, :], s=2, c='black') # plot points
			fig.suptitle(title + '\n' + time_template % t[time])
		return fig, axs,

	mpl.animation.FuncAnimation(fig, ani_run, t.size, ani_init, fargs=(True,), interval=100, repeat=False, blit=False).save(anime_point_file, 'imagemagick')
	fig.clear()
	axs = fig.subplots(nrows=NUM_ROW, ncols=NUM_COL)
	mpl.animation.FuncAnimation(fig, ani_run, t.size, ani_init, fargs=(False,), interval=100, repeat=False, blit=False).save(anime_no_point_file, 'imagemagick')


if __name__ == '__main__':
	mass, xmin, xmax, pmin, pmax, num_points = read_input('input')
	DIM = mass.size
	LOG_FILE, LOG_PIC = 'run.log', 'log.png'
	AVE_FILE, AVE_PIC = 'ave.txt', 'ave.png'
	PRM_FILE, PRM_PIC = 'param.txt', 'param.png'
	PT_FILE, VAL_FILE = 'coord.txt', 'value.txt'
	DEN_PT_TITLE, DEN_PT_ANI = 'Points for Gaussian Process Regression', 'point_density.gif'
	XTR_PT_TITLE, XTR_PT_ANI = 'Points for Optimization', 'point_extra.gif'
	PHS_FILE, PHS_TITLE, PHS_PT_ANI, PHS_NPT_ANI = 'phase.txt', 'Predicted Distribution', 'phase_point.gif', 'phase_no_point.gif'
	ARG_FILE, PHS_ARG_TITLE, PHS_ARG_PT_ANI, PHS_ARG_NPT_ANI = 'argument.txt', 'Predicted Phase Space Distribution', 'phase_arg_point.gif', 'phase_arg_no_point.gif'
	VAR_FILE, VAR_TITLE, VAR_PT_ANI, VAR_NPT_ANI = 'var.txt', 'Variance of Prediction', 'variance_point.gif', 'variance_no_point.gif'

	if DIM == 1: # plot
		# plot error and normalization factor
		t = plot_log(LOG_FILE, LOG_PIC)
		# plot averages
		plot_average(DIM, t, AVE_FILE, AVE_PIC)
		# plot hyperparameters
		plot_param(DIM, t, PRM_FILE, PRM_PIC)
		# plot selected point difference
		plot_point_error(t, num_points, VAL_FILE)
		# animate point evolution
		with open(PHS_FILE, 'r') as pf:
			NUM_GRID = int(np.sqrt(len(pf.readline().split())))
			pf.close()
		x, p = np.linspace(xmin, xmax, NUM_GRID), np.linspace(pmin, pmax, NUM_GRID)
		pt_data = draw_point_anime(DIM, x, p, t, num_points, PT_FILE, VAL_FILE, DEN_PT_TITLE, DEN_PT_ANI, XTR_PT_TITLE, XTR_PT_ANI)
		# load files and animate phase evolution
		phs_data = np.loadtxt(PHS_FILE).reshape((t.size, NUM_TRIG, 2, NUM_GRID, NUM_GRID))
		draw_phase_anime(x, p, t, pt_data, phs_data, PHS_TITLE, PHS_PT_ANI, PHS_NPT_ANI, False)
		draw_phase_anime(x, p, t, pt_data, phs_data, PHS_ARG_TITLE, PHS_ARG_PT_ANI, PHS_ARG_NPT_ANI, False)
		var_data = np.loadtxt(VAR_FILE).reshape((t.size, NUM_TRIG, NUM_GRID, NUM_GRID))
		draw_phase_anime(x, p, t, pt_data, var_data, VAR_TITLE, VAR_PT_ANI, VAR_NPT_ANI, True)
