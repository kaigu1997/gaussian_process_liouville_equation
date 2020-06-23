import numpy as np

# some constants
# bath (atom) mass
mass = 2000.0
# initial position
x0 = -8.0
# box size
xmin = -15.0
xmax = 15.0
# maximum time-step
dt_max = 0.1
# maximum grid-size
dx_max = 0.1
# the number of output time; not exact, but arount that
number_of_output = 50

# log_e(Energy)
ln_energy = float(input())
# initial momentum and deviation
p0 = np.sqrt(2.0 * mass * np.exp(ln_energy))
sigma_p = p0 / 20.0
# total time
total_time = (-x0 - x0) / (p0 / mass)


# calculate output time

# func: calculate a cutoff: to closest 1eN, 2eN, 5eN
# e.g.: 0.11->0.1, 8.2->5, 3626->2000
def cutoff(x):
	# calculate the number of digits, or N
	logx = np.log10(x)
	n = int(logx)
	powx = np.power(10, n)
	# the resume
	resume = logx - n
	# choose the value: 1, 2, 5
	if resume < 0.3: # lg(2)~0.30
		return 2 * powx
	elif resume < 0.7: # lg(5)~0.70
		return 5 * powx
	else:
		return 10 * powx

output_time = cutoff(total_time / number_of_output)


# output
f = open("input", "w")
f.write(f'''mass:
{mass}
x0:
{x0}
p0:
{p0}
Sigma p:
{sigma_p}
xmin:
{xmin}
xmax:
{xmax}
dx:
{dx_max}
output time:
{output_time}
dt:
{dt_max}''')
f.close()