#!/bin/bash

os=$(uname)
scriptDir=$(dirname $0)
cd $scriptDir/text_detection_EAST/inference_local

if [ $os == 'Darwin' ]; then
    # Mac
    cd ../EAST/lanms; make clean; make
    cd ../../inference_local
    python3 ./web_server.py
else
    # SageMaker
    source activate tensorflow_p36
    python3 ./web_server.py
fi

