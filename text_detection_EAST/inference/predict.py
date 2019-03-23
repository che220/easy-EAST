import os, time, datetime, functools, logging, collections, io, base64, hashlib, json
import numpy as np, pandas as pd
import cv2, tensorflow as tf
from tensorflow.python.client import device_lib
from text_detection_EAST.EAST import model
from text_detection_EAST.EAST.nms import non_max_suppression
import text_detection_EAST.inference.utils as utils

# from text_detection_EAST.EAST import lanms

logging.basicConfig(format='%(asctime)s [%(name)s:%(lineno)d] [%(levelname)s] %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
model_dir = '/opt/ml/model'


@functools.lru_cache()
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


@functools.lru_cache()
def get_gpu_list():
    gpus = get_available_gpus()
    if len(gpus) == 0:
        return None

    gpu_list = ",".join([str(i) for i in range(len(gpus))])
    return gpu_list


@functools.lru_cache()
def get_checkpoint_path():
    path = model_dir + '/east_icdar2015_resnet_v1_50_rbox/'
    logger.info('model checkpoint path: %s', path)
    return path


@functools.lru_cache()
def get_predictor():
    gpu_list = get_gpu_list()
    if gpu_list is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        logger.info('set GPU list to: %s', gpu_list)

    checkpoint_path = get_checkpoint_path()
    logger.info('loading model ...')
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                  trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
    logger.info('restore from %s ...', model_path)
    saver.restore(sess, model_path)

    def predictor(img):
        """
        :return: {
            'text_boxes': [
                {
                    'score': ,
                    'x0': ,
                    'y0': ,
                    'x1': ,
                    ...
                    'y3': ,
                }
            ],
            'rtparams': {  # runtime parameters
                'image_size': ,
                'working_size': ,
            },
            'timing': {
                'net': ,
                'restore': ,
                'nms': ,
                'uptime': ,
            }
        }
        """
        start_time = time.time()
        rtparams = collections.OrderedDict()
        rtparams['start_time'] = datetime.datetime.now().isoformat()
        rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
        timer = collections.OrderedDict([('net', 0),
                                         ('restore', 0),
                                         ('nms', 0)
                                         ])

        im_resized, (ratio_h, ratio_w) = resize_image(img)
        rtparams['working_size'] = '{}x{}'.format(im_resized.shape[1], im_resized.shape[0])
        start = time.time()
        # score, geometry = sess.run([f_score, f_geometry],
        # feed_dict={input_images: [im_resized[:,:,::-1]]})
        score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer, in_img=im_resized)
        logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(timer['net'] * 1000,
                                                                          timer['restore'] * 1000,
                                                                          timer['nms'] * 1000))

        scores = None
        if boxes is not None:
            scores = boxes[:, 8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        timer['overall'] = duration
        logger.info('overall time: {}'.format(duration))

        text_boxes = []
        if boxes is not None:
            text_boxes = []
            for box, score in zip(boxes, scores):
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue

                tl = collections.OrderedDict(zip(['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                                                 map(float, box.flatten())))
                tl['score'] = float(score)
                text_boxes.append(tl)
        ret = {
            'text_boxes': text_boxes,
            'rtparams': rtparams,
            'timing': timer,
        }
        return ret

    return predictor


def restore_rectangle_rbox(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x,
                                    5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y,
                                    5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]],
                                 axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose(
            (0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose(
            (0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        # N*4*2
        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


def restore_rectangle(origin, geometry):
    return restore_rectangle_rbox(origin, geometry)


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def resize_image(im, max_side_len=2400):
    """
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(
            max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1,
           nms_thres=0.2, in_img=None):
    """
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :param in_img: image
    :return:
    """
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]

    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4,
                                          geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    logger.info('%d text boxes before nms', text_box_restored.shape[0])
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    # By now, boxes contains 9 columns. The last column is confidence score.
    # The first 8 columns are for quadrangles: (left, top) (right, top), (right, bottom),
    # (left, bottom)
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # Original Code:
    # NOT THIS LINE: boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    # ORIG LINE: boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    # to simplify the case, just use rectangles here
    lefts = np.min(boxes[:, (0, 6)], axis=1)
    rights = np.max(boxes[:, (2, 4)], axis=1)
    tops = np.min(boxes[:, (1, 3)], axis=1)
    bottoms = np.max(boxes[:, (5, 7)], axis=1)
    rects = np.c_[lefts, tops, rights, bottoms]
    # if in_img is not None:
    #     tmp_img = in_img.copy()
    #     for i in range(lefts.shape[0]):
    #         x1, y1, x2, y2 = lefts[i], tops[i], rights[i], bottoms[i]
    #         cv2.rectangle(tmp_img, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
    #     utils.show_image_in_window(tmp_img, 'xy text', (1000, 1000), should_wait=True)

    rects, idxs = non_max_suppression(rects, probs=boxes[:, 8], overlapThresh=nms_thres)
    # The following is getting the rectangles
    # boxes = np.c_[rects[:, 0], rects[:, 1],  # left, top
    #               rects[:, 2], rects[:, 1],  # right, top
    #               rects[:, 2], rects[:, 3],  # right, bottom
    #               rects[:, 0], rects[:, 3],  # left, bottom
    #               boxes[idxs, 8]  # scores
    # ]

    # The following is getting the quadrangles
    boxes = boxes[idxs].copy()

    # back to original code
    timer['nms'] = time.time() - start
    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def decode_image(base64bytes):
    bytes = base64.b64decode(base64bytes)
    logger.info("image bytes: {:,}".format(len(bytes)))
    bio = io.BytesIO(bytes)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    return img


def predict_image(img):
    logger.info('image size: %s', img.shape)
    rst = get_predictor()(img)
    return rst


def predict(input_data):
    img = decode_image(input_data['image'])
    rst = predict_image(img)
    return rst
