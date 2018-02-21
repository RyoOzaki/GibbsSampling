#%% Import parts.
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import patches
import matplotlib.animation as animation
import scipy.stats as stats
from tqdm import trange

#%% Define functions.
def counting(datas, size, out=None):
    if out is None:
        cnt = np.zeros(size, dtype=int)
    else:
        cnt = out
        cnt[:] = 0
    if datas is None:
        return cnt
    for i in range(size):
        cnt[i] = (datas == i).sum()
    return cnt

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

#%% Initialize truth parameters.
K_truth = 3
D = 2
N = 300
ITERATION = 150
pi_truth = np.array([0.7, 0.2, 0.1])
mean_truth = np.array([[3.0, 0.0], [-3.0, 0.0], [0.0, 5.0]])
cov_truth = np.empty((K_truth, D, D))
for i in range(K_truth):
    cov_truth[i] = np.identity(D)


#%% Plot color initializeton.
colors = np.array(["tab:blue", "tab:orange", "tab:pink", "tab:green", "tab:red", "tab:cyan"])

#%% Generate datas.
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
truth_im = []
truth_im.extend(plot_mixingrate(ax_pi, pi_truth, colors=colors[:K_truth]))
#ax.tick_params(labelbottom="off",bottom="off")# Delete x axis.
ax_pi.tick_params(labelleft="off",left="off")# Delete y axis.
ax_pi.set_xlim(0.0, 1.0)
ax_pi.set_ylim(-1.0, 1.0)
ax_data = plt.subplot(gs[1])
for k in range(K_truth):
    truth_im.append(plot_datas(ax_data, datas[hidden_state_truth == k], color=colors[k]))
    truth_im.append(plot_mean(ax_data, mean_truth[k], color=colors[k]))
    truth_im.append(plot_covariance(ax_data, mean_truth[k], cov_truth[k], color=colors[k]))
xlim = ax_data.get_xlim()
ylim = ax_data.get_ylim()
width = xlim[1] - xlim[0]
height = ylim[1] - ylim[0]
ax_data.set_xlim(0.2 * -width + xlim[0], 0.2 * width + xlim[1])
ax_data.set_ylim(0.2 * -height + ylim[0], 0.2 * height + ylim[1])
#plt.show()

#%% Initialize hyper parameters.
K = 4
alpha_0 = np.ones(K)
mu_0 = np.zeros((K, D))
Sigma_0 = np.empty((K, D, D))
for k in range(K):
    Sigma_0[k] = np.identity(D)
nu_0 = np.ones(K) * D
Psi_0 = np.empty((K, D, D))
for k in range(K):
    Psi_0[k] = np.identity(D)

#%% Initialize local parameters.
pi = stats.dirichlet.rvs(alpha_0)[0]
mu = np.empty((K, D))
Sigma = np.empty((K, D, D))
for k in range(K):
    mu[k] = stats.multivariate_normal.rvs(mean=mu_0[k], cov=Sigma_0[k])
    Sigma[k] = stats.invwishart.rvs(df=nu_0[k], scale=Psi_0[k])

#%% Define tmp variables.
k_vec = np.zeros(K)
D_zeros = np.matrix(np.zeros(D))
p_yi = np.empty((N, K))
hidden_state = np.zeros(N, dtype=int)
hidden_state_count = np.zeros(K, dtype=int)

#%% Define animation objects.
ims = []
for i in range(50):
    ims.append(truth_im)

#%% Start Gibbs sampling.
for t in trange(ITERATION):

    # Calculate phase.
    for k in range(K):
        p_yi[:,k] = stats.multivariate_normal.pdf(datas, mean=mu[k], cov=Sigma[k])
    p_yi *= pi

    # Resampling labels.
    for n in range(N):
        k_vec[:] = p_yi[n]
        k_vec /= k_vec.sum()
        hidden_state[n] = np.random.choice(K, p=k_vec)
    counting(hidden_state, K, out=hidden_state_count)

    # Resampling parameters phase.
    # Resampling pi.
    alpha_hat = alpha_0 + hidden_state_count
    pi[:] = stats.dirichlet.rvs(alpha_hat)

    # Resampling mu_k and Sigma_k
    for k in range(K):
        data_k = np.matrix(datas[hidden_state == k])
        if data_k.shape[0] is 0:
            data_k = D_zeros
        M_k = hidden_state_count[k]
        sum_data_k = data_k.sum(axis=0)

        # Resampling Sigma_k.
        tmp = np.matrix(data_k - mu[k])
        Psi_hat = tmp.T.dot(tmp) + Psi_0[k]
        nu_hat = M_k + nu_0[k]
        Sigma[k] = stats.invwishart.rvs(df=nu_hat, scale=Psi_hat)

        # Resampling mu_k.
        Sigma_hat = np.array((M_k * np.matrix(Sigma[k]).I + np.matrix(Sigma_0[k]).I).I)
        mu_hat = np.array((sum_data_k.dot(np.matrix(Sigma[k]).I) + mu_0[k].dot(np.matrix(Sigma_0[k]).I)).dot(Sigma_hat))[0]
        mu[k] = stats.multivariate_normal.rvs(mean=mu_hat, cov=Sigma_hat)

    # Save animation phase.
    tmp_im = []
    tmp_im.extend(plot_mixingrate(ax_pi, pi, colors=colors[:K]))
    tmp_im.append(ax_data.annotate("ITERATION : %3d" % (t+1,), xy=(0.7,0.08), xycoords='axes fraction', fontsize=15, bbox=dict(boxstyle="round", fc="white"), zorder=3))
    for k in range(K):
        if hidden_state_count[k] != 0:
            data_k = datas[hidden_state == k]
            tmp_im.append(plot_datas(ax_data, data_k, color=colors[k]))
        tmp_im.append(plot_mean(ax_data, mu[k], color=colors[k]))
        tmp_im.append(plot_covariance(ax_data, mu[k], Sigma[k], color=colors[k]))
    ims.append(tmp_im)

#%% Add animation interval
for i in range(50):
    ims.append(ims[-1])

#%% Plot phase.
ax_pi.set_title("Gibbs sampling for GMM.")
anim = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=0, blit=True)
#plt.show()
anim.save('GMM_Gibbs.gif', writer='imagemagick', fps=10)
