from skimage import feature, transform
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import IFrame
import warnings
import http.server
import socketserver
import asyncio
import websockets
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common
import cv2

def run_websocket_server(websocket_handler, port=1234):
    print(f'Starting websocket server on port {port}')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(websocket_handler, "127.0.0.1", port)
    loop.run_until_complete(start_server)
    loop.run_forever()

def run_http_server(port = 8080):
    Handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print("serving at port", port)
        httpd.serve_forever()

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

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def compare_images(img1, img2):
    if not is_port_in_use(8000):
        warnings.warn('The server seems not to be running pls start with: python -m http.server')
        return
    
    # save the images
    abs_max = np.percentile(np.abs(img1), 100)
    abs_min = abs_max
    plt.imsave('./html/first.png', img1, cmap='Greys', vmin=-abs_min, vmax=abs_max)
    
    colormap_transparent = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['blue','red'], 256)
    colormap_transparent._init() # create the _lut array, with rgba values

    alphas = np.linspace(0, 2.0 * np.pi, colormap_transparent.N+3)
    colormap_transparent._lut[:,-1] = list(map(lambda x : np.clip(np.cos(x) + 0.5, 0.0, 1.0), alphas))
    
    plt.imsave('./html/second.png', img2, cmap=colormap_transparent)
    
    return IFrame(src='http://0.0.0.0:8000/html/diffViewer.html', width=700, height=600)
    
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
    
    
def plot_pose(image, humans, heatMat):
    image_result = TfPoseEstimator.draw_humans(image, humans, imgcopy=True)

    fig = plt.figure(figsize=(50, 25))
    a = fig.add_subplot(2, 1, 1)
    a.set_title('Result')
    plt.imshow(cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB))

    bgimg = cv2.cvtColor(image_result.astype(np.uint8), cv2.COLOR_BGR2RGB)
    bgimg = cv2.resize(bgimg, (heatMat.shape[1], heatMat.shape[0]), interpolation=cv2.INTER_AREA)

    # show network output
    a = fig.add_subplot(2, 2, 2)
    plt.imshow(bgimg, alpha=0.5)
    tmp = np.amax(heatMat[:, :, :-1], axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    _ = plt.colorbar()
    
def get_humans_as_lines(humans, image_h, image_w):
    centers = {}
    lines = []
    for human in humans:
        # draw point
        for i in range(common.CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), image_h - int(body_part.y * image_h + 0.5))
            centers[i] = center
            
        # draw line
        for pair_order, pair in enumerate(common.CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue

            
            line = [centers[pair[0]], centers[pair[1]]]
            lines.append(line)
    return lines

def plot_human_lines(lines, axis, color = 'r', linestyle='-', label='human'):
    for line in lines:
        x = [line[0][0], line[1][0]]
        y = [line[0][1], line[1][1]]
        axis.plot(x, y, color=color, linestyle=linestyle, label=label, marker='o')
        
        
def is_same_image(a, b):
    return np.sum(np.abs(a-b))