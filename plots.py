from matplotlib import pyplot as plt
import numpy as np

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 6))

ax = axs[0,0]

x = np.linspace(2,5,10)
y = x**2 + x + 1 + np.random.rand(len(x)) # some random data

z = np.linspace(2,5,10)
t = z**2 + z + 1 + np.random.rand(len(z)) + 2*z # some random data

ax.set_title("title")
ax.plot(x, y , label="plot1")
ax.plot(z, t , label="plot2")
ax.legend(fontsize=8)
ax.set_xlabel("n_wires")#, fontsize=16)
ax.set_ylabel("time (s)")#, fontsize=16)
ax.set_yscale("linear")
ax.set_xscale("linear")

ax = axs[1,0]

ax.set_title("title")
ax.plot(x, y , "x--", label="legend description")
ax.legend(fontsize=8)
ax.set_xlabel("nx")#, fontsize=16)
ax.set_ylabel("y $\\theta$ also latex no problem")#, fontsize=16)
ax.set_yscale("log")
ax.set_xscale("log")


# ax = axs[0,0]

# x = np.linspace(2,5,10)
# y = x**2 + x + 1 + np.random.rand(len(x)) # some random data

# ax.set_title("title")
# ax.plot(x, y , "x--", label="legend description")
# ax.legend(fontsize=8)
# ax.set_xlabel("n_wires")#, fontsize=16)
# ax.set_ylabel("time (s)")#, fontsize=16)
# ax.set_yscale("linear")
# ax.set_xscale("linear")

# ax = axs[1,0]

# ax.set_title("title")
# ax.plot(x, y , "x--", label="legend description")
# ax.legend(fontsize=8)
# ax.set_xlabel("nx")#, fontsize=16)
# ax.set_ylabel("y $\\theta$ also latex no problem")#, fontsize=16)
# ax.set_yscale("log")
# ax.set_xscale("log")

plt.savefig("savefiguresomewhere.png", dpi=200)

#--------------------------------------------------------------------------------------


# import pandas as pd
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from matplotlib import rc
# from matplotlib import gridspec
# import cmath
# import scipy as scipy
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import ListedColormap
# from ast import literal_eval    

# #plt.style.use('/Users/paolostornati/Phd/phdthesis.mpltstyle')

# niceblack="#262626"
# from matplotlib.colors import ListedColormap
# import palettable
# from palettable.cartocolors.sequential import DarkMint_7
# cmap = ListedColormap(DarkMint_7.mpl_colors)
# colors_mint = cmap.colors
# # from palettable.colorbrewer.sequential import Blues_8

# cmap = ListedColormap(palettable.cmocean.sequential.Amp_20.mpl_colors)
# reds = cmap.colors

# cmap = ListedColormap(palettable.cmocean.diverging.Balance_14.mpl_colors)
# balance14 = cmap.colors

# cmap = ListedColormap(palettable.matplotlib.Inferno_20.mpl_colors)
# inferno = cmap.colors

# plt.savefig("savefiguresomewhere2.png", dpi=200)