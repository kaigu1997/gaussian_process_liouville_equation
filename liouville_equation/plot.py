import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

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
dx = (x[len(x)-1] - x[0]) / (len(x) - 1)
dp = (p[len(p)-1] - p[0]) / (len(p) - 1)
xv, pv = np.meshgrid(x, p) # transform to vector for plotting
# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
xmax00 = 0.2
xmax11 = 0.05
levels00 = MaxNLocator(nbins=15).tick_values(-xmax00, xmax00) # color region
levels11 = MaxNLocator(nbins=15).tick_values(-xmax11, xmax11) # color region
cmap = plt.get_cmap('seismic') # the kind of color: red-white-blue
#norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True) # the mapping rule

# initialize the plot
fig, (ax0, ax1) = plt.subplots(nrows=2)

# set time text
time_template = 'time = %da.u.'

# initialization: blank page
def init():
    # clear, then set x/y label, title of subplot, and colorbar
    ax0.clear()
    ax0.set_xlabel('x')
    ax0.set_ylabel('p')
    ax0.set_title(r'$\rho_{00}$')
    cf0 = ax0.contourf(xv, pv, np.zeros((len(x), len(p))), levels=levels00, cmap=cmap)
    fig.colorbar(cf0, extend='both', ax=ax0)
    ax1.clear()
    ax1.set_xlabel('x')
    ax1.set_ylabel('p')
    ax1.set_title(r'$\rho_{11}$')
    cf1 = ax1.contourf(xv, pv, np.zeros((len(x), len(p))), levels=levels11, cmap=cmap)
    cf1.set_clim(-xmax11, xmax11)
    fig.colorbar(cf1, extend='both', ax=ax1)
    # figure settings: make them closer, title to be time
    fig.suptitle('')
    return fig, ax0, ax1,

# animation: each timestep in phase.txt
def ani(i):
    # get data, in rhoi_real
    file = open('phase.txt', 'r')
    # old data
    for j in range(i - 1):
        file.readline() # rho[0][0]
        file.readline() # rho[0][1]
        file.readline() # rho[1][0]
        file.readline() # rho[1][1]
        file.readline() # blank line
    # new data
    rho0 = np.array(file.readline().split(), dtype=float)
    for j in range(len(rho0)//2):
        rho0[j] = rho0[2 * j]
    rho0_real = rho0[0:len(rho0)//2].reshape(len(p),len(x)).T
    file.readline() # rho[0][1]
    file.readline() # rho[1][0]
    rho1 = np.array(file.readline().split(), dtype=float)
    for j in range(len(rho1)//2):
        rho1[j] = rho1[2 * j]
    rho1_real = rho1[0:len(rho1)//2].reshape(len(p),len(x)).T
    file.close()

    # print contourfs
    cf0 = ax0.contourf(xv, pv, rho0_real, levels=levels00, cmap=cmap)
    cf1 = ax1.contourf(xv, pv, rho1_real, levels=levels11, cmap=cmap)
    fig.suptitle(time_template % t[i])
    return fig, ax0, ax1,

# make the animation
ani = animation.FuncAnimation(fig, ani, len(t), init, interval=10000//(t[1]-t[0]), repeat=False, blit=False)
# show
ani.save('phase.gif','imagemagick')
# plt.show()


# plot population evolution
def calc_pop():
    file = open('phase.txt', 'r')
    ppl = [[],[]]
    for i in range(len(t)):
        ppl[0].append(sum(np.array(file.readline().split(), dtype=float)) * dx * dp)
        file.readline() # rho[0][1]
        file.readline() # rho[1][0]
        ppl[1].append(sum(np.array(file.readline().split(), dtype=float)) * dx * dp)
        file.readline() # blank line
    file.close()
    return ppl

plt.clf()
ppl = calc_pop()
plt.plot(t, ppl[0], color='r', label='Population[0]')
plt.plot(t, ppl[1], color='b', label='Population[1]')
plt.legend(loc = 'best')
plt.xlim((t[0],t[len(t)-1]))
plt.ylim((0,1))
plt.savefig('phase.png')
