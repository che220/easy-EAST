import os
import logging
import json
import cv2
import pandas as pd
import numpy as np
from text_detection_EAST.inference import predict
from text_detection_EAST.EAST import model

logging.basicConfig(format='%(asctime)s [%(name)s:%(lineno)d] [%(levelname)s] %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 5000)
pd.set_option('max_columns', 600)
np.set_printoptions(linewidth=5000)  # print out all values, regardless length

data_dir = '/data/input/training'


def define_local_dirs():
    global data_dir
    data_dir = 'D:/data_cache/' + model.MODEL_NAME
    logger.info('data dir is set to %s', data_dir)

    # file_path = os.path.dirname(os.path.abspath(__file__))
    # predict.model_dir = file_path + '/../trained_models'
    predict.model_dir = 'D:/data_cache/EAST_models'
    logger.info('model dir is set to %s', predict.model_dir)
    cp_path = predict.get_checkpoint_path()
    if not os.path.exists(cp_path):
        raise RuntimeError('cannot find checkpoint path {}'.format(cp_path))


def draw_illu(illu, rst):
    for t in rst['text_boxes']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'], t['y2'], t['x3'], t['y3']],
                     dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 0, 0), thickness=2)
    return illu


def save_result(img, rst, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    # save input image
    output_path = dir_path + '/input.png'
    cv2.imwrite(output_path, img)

    # save illustration
    output_path = dir_path + '/output.png'
    cv2.imwrite(output_path, draw_illu(img.copy(), rst))  # draw text boxes here!!!

    # save json data
    output_path = dir_path + '/result.json'
    with open(output_path, 'w') as f:
        json.dump(rst, f)

    logger.info('results are saved into %s', dir_path)
    return rst
