'''
model performance on astroboy:

GPU: GPU:0 with 7468 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)

Receipts:

2018-09-06 18:25:05,718 [inference.predict:99] [INFO] net 156ms, restore 2ms, nms 46ms
2018-09-06 18:25:05,718 [inference.predict:110] [INFO] overall time: 0.21676325798034668

2018-09-06 18:25:27,608 [inference.predict:99] [INFO] net 151ms, restore 2ms, nms 61ms
2018-09-06 18:25:27,608 [inference.predict:110] [INFO] overall time: 0.23383116722106934

IRS 1099-MISC:

2018-09-06 18:29:49,629 [inference.predict:99] [INFO] net 417ms, restore 2ms, nms 308ms
2018-09-06 18:29:49,629 [inference.predict:110] [INFO] overall time: 0.7453327178955078

'''

import os, sys, base64, logging, io, cv2, numpy as np, uuid
from flask import Flask, request, jsonify, render_template

logging.basicConfig(format='%(asctime)s [%(name)s:%(lineno)d] [%(levelname)s] %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(path); path = os.path.dirname(path)
logger.info('add %s to PYTHONPATH', path)
sys.path.append(path)

from text_detection_EAST.inference import predict
from text_detection_EAST.inference_local import local_env

local_env.define_local_dirs()
app = Flask(__name__)

@app.route('/')
def index():
    session_id = 'dummy_session_id'
    print('session id:', session_id)
    return render_template('index.html', session_id=session_id)

@app.route('/', methods=['POST'])
def image_post():
    '''
    from index.html. POST a file

    :return:
    '''
    bio = io.BytesIO()
    request.files['image'].save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    rst = predict.predict_image(img)
    session_id = save_results(img, rst)
    return render_template('index.html', session_id=session_id)

@app.route('/invocations', methods=['POST'])
def invocations():
    '''
    POST base64-encoded image binary data

    :return:
    '''
    request.get_json(force=True)
    img = predict.decode_image(request.json['image'])
    rst = predict.predict_image(img)

    img_out = local_env.draw_illu(img.copy(), rst)
    buf = cv2.imencode('.png', img_out)[1].tostring()
    rst['output_image'] = base64.b64encode(buf).decode('utf-8')
    return jsonify(rst)

def save_results(img, rst):
    session_id = str(uuid.uuid1())
    logger.info('request session id: %s', session_id)

    out_path = 'static/results'
    out_dir = out_path + '/{}'.format(session_id)
    os.makedirs(out_dir, exist_ok=True)
    local_env.save_result(img, rst, out_dir)
    return session_id

if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5001
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    local_env.define_local_dirs()
    app.run(host=host, port=port)
    print()
