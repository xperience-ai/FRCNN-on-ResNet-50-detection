import argparse
import os

from detection.demo.utils import show_result
from detection.model.api import init_detector, inference_detector


def run_detection(img, show_res=True):
    config_path = 'configs/'
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    config_file = config_path + 'faster_rcnn_r50_c4_2x.py'
    model = init_detector(config_file, device='cuda:0')
    result = inference_detector(model, img)
    return show_result(img, result, model.CLASSES, score_thr=0.7, font_scale=.6, thickness=1, show_mask=False, show=show_res)


def main():
    parser = argparse.ArgumentParser(description='Faster-RCNN on ResNet-50')
    parser.add_argument('--path_img', type=str, default='data/cat.jpg')
    global args
    args = parser.parse_args()
    img = args.path_img
    run_detection(img)


if __name__ == '__main__':
    main()
