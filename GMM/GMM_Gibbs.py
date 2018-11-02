#%% Import.
# %matplotlib inline
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import patches
import scipy.stats as stats
from tqdm import trange

#%% Functions.
def rad2deg(rad):
    return rad * 180.0 / np.pi

def plot_mixingrate(axes, mixing_rate, colors=None):
    size = mixing_rate.shape[0]
    arg = np.argsort(mixing_rate)[::-1]
    cumsum_rev = np.cumsum(mixing_rate[arg])[::-1]
    if colors is not None:
        colors = colors[arg][::-1]
    return axes.barh([0] * size, cumsum_rev, color=colors, height=2.0).get_children()

def plot_mean(axes, mean, color=None, mark="o"):
    return axes.scatter(mean[0], mean[1], marker=mark, c=color, zorder=1)

def plot_covariance(axes, mean, covariance, scale=3.0, color=None):
    la, v = np.linalg.eig(covariance)
    std = np.sqrt(la)
    angle = rad2deg(np.arctan2(v[1,0], v[0,0]))
    e = patches.Ellipse((mean[0], mean[1]), 2*std[0]*scale, 2*std[1]*scale, angle=angle, linewidth=1, fill=False, color=color, zorder=2)
    axes.add_artist(e)
    return e

def plot_datas(axes, datas, color=None, mark="+"):
    return axes.scatter(datas[:,0], datas[:,1], marker=mark, c=color, zorder=1)

#%% Make a figure dir and delete old files.
if not Path("fig").exists():
    Path("fig").mkdir()
imgs = Path("fig").glob("*.png")
for img in imgs:
    img.unlink()

#%% Truth parameters.
K_truth = 3
D = 2
N = 3000
ITERATION = 150
pi_truth = np.array([0.7, 0.2, 0.1])
mean_truth = np.array([[3.0, 0.0], [-3.0, 0.0], [0.0, 5.0]])
cov_truth = np.empty((K_truth, D, D))
for i in range(K_truth):
    cov_truth[i] = np.identity(D)


#%% Plot color initializeton.
colors = np.array(["tab:blue", "tab:orange", "tab:pink", "tab:green", "tab:red", "tab:cyan"])

#%% Observations.
datas = np.empty((N, D))
hidden_state_truth = np.random.choice(K_truth, size=N, p=pi_truth)
for n in range(N):
    k = hidden_state_truth[n]
    datas[n] = stats.multivariate_normal.rvs(mean=mean_truth[k], cov=cov_truth[k])

#%% Plot datas with truth label and get animation objects.
plt.rcParams["font.size"] = 18
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6])
ax_pi = plt.subplot(gs[0])
plot_mixingrate(ax_pi, pi_truth, colors=colors[:K_truth])
#ax.tick_params(labelbottom="off",bottom="off")# Delete x axis.
ax_pi.tick_params(labelleft="off", left="off")# Delete y axis.
ax_pi.set_xlim(0.0, 1.0)
ax_pi.set_ylim(-1.0, 1.0)
ax_data = plt.subplot(gs[1])
for k in range(K_truth):
    plot_datas(ax_data, datas[hidden_state_truth == k], color=colors[k])
    plot_mean(ax_data, mean_truth[k], color=colors[k])
    plot_covariance(ax_data, mean_truth[k], cov_truth[k], color=colors[k])
xlim = ax_data.get_xlim()
ylim = ax_data.get_ylim()
width = xlim[1] - xlim[0]
height = ylim[1] - ylim[0]
xlim = (0.2 * -width + xlim[0], 0.2 * width + xlim[1])
ylim = (0.2 * -height + ylim[0], 0.2 * height + ylim[1])
ax_data.set_xlim(xlim)
ax_data.set_ylim(ylim)
#plt.show()
plt.savefig("fig/truth.png")

#%% Hyper parameters.
K = 4
alpha_0 = np.ones(K)
mu_0 = np.zeros((K, D))
lambda_0 = np.ones(K)
nu_0 = np.ones(K) * D
Psi_0 = np.empty((K, D, D))
for k in range(K):
    Psi_0[k] = np.identity(D)

#%% Local parameters.
pi = stats.dirichlet.rvs(alpha_0)[0]
mu = np.empty((K, D))
Sigma = np.empty((K, D, D))
for k in range(K):
    Sigma[k] = stats.invwishart.rvs(df=nu_0[k], scale=Psi_0[k])
    mu[k] = stats.multivariate_normal.rvs(mean=mu_0[k], cov=(Sigma[k] / lambda_0[k]))

#%% Temporary variables.
p_xi = np.empty((N, K))

#%% Latent variables
hidden_state = np.zeros(N, dtype=int)

#%% Gibbs sampling iteration.
for t in trange(ITERATION):

    # Calculate phase.
    for k in range(K):
        p_xi[:,k] = stats.multivariate_normal.pdf(datas, mean=mu[k], cov=Sigma[k])
    p_xi *= pi

    # Resampling labels.
    for n in range(N):
        pi_ast = p_xi[n] / p_xi[n].sum()
        hidden_state[n] = np.random.choice(K, p=pi_ast)
    m = np.bincount(hidden_state, minlength=K)

    # Resampling pi.
    alpha_ast = alpha_0 + m
    pi[:] = stats.dirichlet.rvs(alpha_ast)

    # Resampling mu_k and Sigma_k
    for k in range(K):
        if m[k] == 0:
            data_k = np.matrix(np.zeros(D))
        else:
            data_k = np.matrix(datas[hidden_state == k])

        # Calculate Temporary variables.
        mean_of_data_k = data_k.mean(axis=0)
        S_k = np.dot((data_k - mean_of_data_k).T, data_k - mean_of_data_k)

        # Resampling Sigma_k.
        nu_ast = m[k] + nu_0[k]
        Psi_ast = S_k + Psi_0[k] + (m[k] * lambda_0[k])/(m[k] + lambda_0[k]) * np.dot((mean_of_data_k - mu_0[k]).T, mean_of_data_k - mu_0[k])
        Sigma[k] = stats.invwishart.rvs(df=nu_ast, scale=Psi_ast)

        # Resampling mu_k.
        lambda_ast = m[k] + lambda_0[k]
        mu_ast = ((m[k] * mean_of_data_k + lambda_0[k] * mu_0[k]) / (m[k] + lambda_0[k])).A1
        mu[k] = stats.multivariate_normal.rvs(mean=mu_ast, cov=(Sigma[k] / lambda_ast))

    # Save animation phase.
    plt.cla()
    plot_mixingrate(ax_pi, pi, colors=colors[:K])
    for k in range(K):
        if m[k] != 0:
            data_k = datas[hidden_state == k]
            plot_datas(ax_data, data_k, color=colors[k])
        plot_mean(ax_data, mu[k], color=colors[k])
        plot_covariance(ax_data, mu[k], Sigma[k], color=colors[k])
    ax_data.set_xlim(xlim)
    ax_data.set_ylim(ylim)
    # plt.pause(0.1)
    plt.savefig("fig/frame{0:03d}.png".format(t+1))

plt.cla()
plot_mixingrate(ax_pi, pi, colors=colors[:K])
for k in range(K):
    if m[k] != 0:
        data_k = datas[hidden_state == k]
        plot_datas(ax_data, data_k, color=colors[k])
    plot_mean(ax_data, mu[k], color=colors[k])
    plot_covariance(ax_data, mu[k], Sigma[k], color=colors[k])
ax_data.set_xlim(xlim)
ax_data.set_ylim(ylim)
# plt.show()
plt.savefig("fig/final.png")
