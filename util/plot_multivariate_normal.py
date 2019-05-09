#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

#%% functions
def rad2deg(rad):
    return rad * 180.0 / np.pi

def plot_mean(mean, color=None, marker="o", axes=None):
    if axes is None:
        axes = plt.gca()
    axes.scatter(*mean, marker=marker, c=color, zorder=1)

def plot_cov(mean, cov, scale=3.0, color=None, axes=None):
    if axes is None:
        axes = plt.gca()
    la, v = np.linalg.eig(cov)
    std = np.sqrt(la)
    angle = rad2deg(np.arctan2(*v[1,:]))
    e = patches.Ellipse(mean, *(2*std*scale), angle=angle, linewidth=1, fill=False, color=color, zorder=2)
    axes.add_artist(e)

def plot_multivariate_normal(mean, cov, marker="o", scale=3.0, color=None, axes=None):
    if axes is None:
        axes = plt.gca()
    # plot cov
    la, v = np.linalg.eig(cov)
    std = np.sqrt(la)
    angle = rad2deg(np.arctan2(*v[1,:]))
    e = patches.Ellipse(mean, *(2*std*scale), angle=angle, linewidth=1, fill=False, color=color, zorder=2)
    axes.add_artist(e)
    # plot mean
    axes.scatter(*mean, marker=marker, c=color, zorder=1)

def plot_datas(datas, color=None, marker="+", axes=None):
    if axes is None:
        axes = plt.gca()
    axes.scatter(datas[:,0], datas[:,1], marker=marker, c=color, zorder=1)

#%% sample code
if __name__ == "__main__":
    from scipy.stats import multivariate_normal
    mean = np.zeros(2)
    cov = np.identity(2,2) * 20
    cov[0, 0] = 10
    cov[0, 1] = -8
    cov[1, 0] = -8
    data = multivariate_normal.rvs(mean, cov=cov, size=100)
    plot_datas(data, color="green")
    plot_multivariate_normal(mean, cov, color="tab:red")

    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    plt.show()
