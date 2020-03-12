from skimage import feature, transform
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from tf_pose.estimator import TfPoseEstimator
from tf_pose import common
import io
from utils import get_humans_as_lines
from matplotlib.lines import Line2D
import cv2


def plot(data, xi=None, cmap='RdBu_r', axis=plt, percentile=100, dilation=3.0, alpha=0.8):
    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, data.shape[1], dx)
    yy = np.arange(0.0, data.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)
    overlay = None
    if xi is not None:
        # Compute edges (to overlay to heatmaps later)
        xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
        in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(data), percentile)
    abs_min = abs_max

    if len(data.shape) == 3:
        data = np.mean(data, 2)
    axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
    axis.axis('off')
    return axis


def plot_vector_field(U, V, bgimg, axis, figure):
    axis.imshow(bgimg, alpha=1)
    X, Y = np.meshgrid(range(0, bgimg.shape[1]), range(0, bgimg.shape[0]))

    color = np.sqrt(np.square(U) + np.square(V))

    # normalize
    #U /= color
    #V /= color

    colormap_transparent = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['blue','red'], 256)
    colormap_transparent._init() # create the _lut array, with rgba values

    alphas = np.linspace(0, 1.0, colormap_transparent.N+3)
    colormap_transparent._lut[:,-1] = alphas # list(map(lambda x : 1.0 if x > 0.3 else 0.0, alphas))

    heat_image = axis.quiver(X, Y, U, V, color, cmap=colormap_transparent, scale=40)
    #heat_image = ax.imshow(V, cmap=plt.cm.hot, alpha=0.5)


    axis.set_title('PAF')
    figure.colorbar(heat_image, ax=axis, shrink=1.0)

    return axis
    #ax = fig.add_subplot(1, 2, 2)
    #heat_image = ax.imshow(color)
    #fig.colorbar(heat_image, ax=ax, shrink=1.0)


def plot_pose(image, humans, heatMat=None):
    image_result = TfPoseEstimator.draw_humans(image, humans, imgcopy=True)

    fig = plt.figure(figsize=(50, 25))
    a = fig.add_subplot(2, 1, 1)
    a.set_title('Result')
    plt.imshow(cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB))

    bgimg = cv2.cvtColor(image_result.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if heatMat is not None:
        bgimg = cv2.resize(bgimg, (heatMat.shape[1], heatMat.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        bgimg = cv2.resize(bgimg, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

    # show network output
    if heatMat is not None:
        a = fig.add_subplot(2, 2, 2)
        plt.imshow(bgimg, alpha=0.5)
        tmp = np.amax(heatMat[:, :, :-1], axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        _ = plt.colorbar()

def plot_humans_lines(humans, axis, color = 'r', linestyle='-', label='human'):
    for human in humans:
        plot_human_lines(human, axis, color, linestyle, label)
        
def plot_human_lines(lines, axis, color = 'r', linestyle='-', label='human'):
    for line in lines:
        x = [line[0][0], line[1][0]]
        y = [line[0][1], line[1][1]]
        axis.plot(x, y, color=color, linestyle=linestyle, label=label, marker='o')


def gen_plot(human_source, human_target, human_adv):
    source_lines = get_humans_as_lines(human_source, 400, 450)
    target_lines = get_humans_as_lines(human_target, 400, 450)
    adv_lines = get_humans_as_lines(human_adv, 400, 450)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    plot_human_lines(source_lines, ax, color='r', linestyle='-', label='source')
    plot_human_lines(target_lines, ax, color='g', linestyle='-', label='target')
    plot_human_lines(adv_lines, ax, color='b', linestyle='--', label='adv')

    legend_elements = [Line2D([0], [0], color='r', label='source'),
                       Line2D([0], [0], color='g', label='target'),
                       Line2D([0], [0], color='b', label='adverserial')]
    ax.legend(handles=legend_elements, loc='best',  prop={'size': 20})
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def gen_plot_universal_noise(noise, index=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.clip(noise, 0, 255) / 255)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    if index != None:
        plt.savefig(f'../logs/gif/{i}.png', format='png')
    return buf