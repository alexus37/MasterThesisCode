import argparse
import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger('TfPoseEstimatorRun').setLevel(logging.ERROR)
logging.getLogger('DeepExplain').setLevel(logging.ERROR)
logging.getLogger('TfPoseEstimator').setLevel(logging.ERROR)


from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from plot_utils import plot_pose
from matplotlib import pyplot

def compute_pose(image_path,  width, height, model, resize_out_ratio, show_result):
    image = common.read_imgfile(image_path, width, height)
    estimator = TfPoseEstimator(get_graph_path(model), target_size=(width, height), trt_bool=False)
    humans = estimator.inference(image, resize_to_default=(width > 0 and height > 0), upsample_size=resize_out_ratio)

    if len(humans) == 0:
        print('Nothing was detected')
        return

    if len(humans[0].body_parts) != 18:
        print(f'Partially detected. {len(humans[0].body_parts)} joints found.')
    else:
        print('Full detection!')

    if show_result:
        plot_pose(image, humans)
        pyplot.show()

if __name__ == "__main__":
    args = argparse.ArgumentParser('Compute pose of given image')

    args.add_argument('--resize_out_ratio', default=2.0, type=float)
    args.add_argument('--show_result', default=False, type=bool)
    args.add_argument('--width', default=432, type=int)
    args.add_argument('--height', default=368, type=int)
    args.add_argument('--model', default='cmu', type=str)
    args.add_argument('--image_path', type=str, required=True)

    current_args = vars(args.parse_args())

    # TODO: check if destructing is posible
    if not os.path.isfile(current_args['image_path']):
        print ("Image does not exist")
    else:
        compute_pose(**current_args)
