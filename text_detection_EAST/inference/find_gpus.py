from text_detection_EAST.inference import predict

if __name__ == '__main__':
    gpus = predict.get_available_gpus()
    print(gpus)

    gpus = predict.get_gpu_list()
    print(len(gpus))
    if gpus is None:
        print('NONE')
    else:
        print(gpus)

