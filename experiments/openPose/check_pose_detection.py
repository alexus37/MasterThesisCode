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

def compute_pose(image_path,  width, height, model, resize_out_ratio, show_result):
    image = common.read_imgfile(image_path, width, height)
    estimator = TfPoseEstimator(get_graph_path(model), target_size=(width, height), trt_bool=False)
    humans = estimator.inference(image, resize_to_default=(width > 0 and height > 0), upsample_size=resize_out_ratio)

    if len(humans) == 0:
        print('Nothing was detected')
        return
    # TODO: check size is correct
    if len(humans[0].body_parts) != 19:
        print(f'Partially detected. {len(humans[0].body_parts)} joints found.')
    else:
        print('Full detection!')

    # TODO: check this works
    if show_result:
        plot_pose(image, humans, estimator.heatMat)

if __name__ == "__main__":
    args = argparse.ArgumentParser('Compute pose of given image')

    args.add_argument('--resize_out_ratio', default=2.0, type=float)
    args.add_argument('--show_result', default=False, type=bool)
    args.add_argument('--width', default=432, type=int)
    args.add_argument('--height', default=368, type=int)
    args.add_argument('--model', default='cmu', type=str)
    args.add_argument('--image_path', type=str, required=True)

    current_args = args.parse_args()

    # TODO: check if file existsif os.
    # TODO: check if destructing is posible
    compute_pose(
        image_path=current_args.image_path,
        width=current_args.width,
        height=current_args.height,
        model=current_args.model,
        resize_out_ratio=current_args.resize_out_ratio,
        show_result=current_args.show_result
    )
