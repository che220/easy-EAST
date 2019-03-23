import os
import logging
import sys
import cv2
import hashlib
import base64
import numpy as np
from text_detection_EAST.inference import predict
from text_detection_EAST.inference.utils import show_image_in_window

from text_detection_EAST.inference_local import local_env
from tkinter import Tk, Frame, Button, BOTH, X, messagebox
from tkinter.filedialog import askopenfilename

logger = logging.getLogger(__name__)


def process_image(in_img_path):
    input_data = {}
    with open(in_img_path, 'rb') as fin:
        input_data['image'] = base64.b64encode(fin.read())
    rst = predict.predict(input_data)

    img = cv2.imread(in_img_path, cv2.IMREAD_UNCHANGED)
    boxes = rst['text_boxes']
    for i in range(len(boxes)):
        rect = boxes[i]
        x0, y0 = rect['x0'], rect['y0']
        x1, y1 = rect['x1'], rect['y1']
        x2, y2 = rect['x2'], rect['y2']
        x3, y3 = rect['x3'], rect['y3']
        pts = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], np.int32)
        pts = pts.reshape(((-1,1,2)))
        cv2.polylines(img, [pts], True, (255, 0, 0), thickness=2)
    show_image_in_window(img, in_img_path, (1000, 1000), should_wait=True)

    md5 = hashlib.md5()
    md5.update(input_data['image'])
    img_hash = md5.hexdigest()
    logger.info('image hash: %s', img_hash)

    out_dir = predict.model_dir + '/results/{}'.format(img_hash)
    os.makedirs(out_dir, exist_ok=True)
    local_env.save_result(img, rst, out_dir)
    return out_dir + '/output.png'


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.master.title('Demo Image Processing')

        Button(self, text='Select Image File', command=self._select_image_file).pack(fill=X,
                                                                                     padx=10,
                                                                                     pady=10)

        Button(self, text='Exit', command=self._client_exit).pack(fill=X, padx=10, pady=10)
        self.pack(fill=BOTH)

    @staticmethod
    def _select_image_file():
        in_img_path = askopenfilename()
        if in_img_path is None or in_img_path == '':
            return

        process_image(in_img_path)

    @staticmethod
    def _client_exit():
        if messagebox.askyesno("Please Verify", "Do you really want to exit?"):
            exit(0)


if __name__ == '__main__':
    local_env.define_local_dirs()

    img_path = None
    root = None
    try:
        root = Tk()
        root.geometry('400x300+0+0')
        app = Window(root)
        root.mainloop()
        exit(0)
    except Exception as exc:
        root = None
        if len(sys.argv) >= 2:
            img_path = sys.argv[1]
        else:
            logger.error('Cannot run tkinter. Abort')
            exit(1)

    process_image(img_path)
