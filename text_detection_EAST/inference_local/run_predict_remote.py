import os, sys, re, logging, cv2, requests, base64
from text_detection_EAST.inference_local import local_env
from text_detection_EAST.inference import predict

logger = logging.getLogger(__name__)

def usage():
    logger.info("usage: %s [image path] <ip:port>", sys.argv[0])

if __name__ == '__main__':
    if '-h' in sys.argv:
        usage()
        exit(0)

    if len(sys.argv) < 2:
        usage()
        exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        logger.error('cannot find file %s', img_path)
        exit(2)

    host = '127.0.0.1'
    port = 5001
    if len(sys.argv) > 2:
        flds = re.split(':', sys.argv[2])
        if len(flds) == 2:
            host = flds[0]
            port = int(flds[1])
        else:
            host = flds[0]
    url = 'http://{host}:{port}/invocations'.format(host=host, port=port)
    logger.info('URL: %s', url)
    logger.info('image file: %s', img_path)

    img_data = {}
    with open(img_path, 'rb') as fin:
        img_data['image'] = base64.b64encode(fin.read()).decode('utf-8')

    res = requests.post(url, json=img_data)
    resp_data = res.json()
    logger.info('\n%s', resp_data)

    img_out = predict.decode_image(resp_data['output_image'])
    logger.info('output image: %s', img_out.shape)
    title = 'Processed Image'
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    width = 1000
    height = int(img_out.shape[0] * width/float(img_out.shape[1]))
    cv2.resizeWindow(title, width, height)
    cv2.imshow(title, img_out)
    cv2.waitKey(0) & 0xFF # for 64-bit machine
    cv2.destroyAllWindows()
