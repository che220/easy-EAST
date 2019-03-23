#!/bin/bash

function usage()
{
    echo "usage: $0 [web server ip:port] [image path]"
    echo '    image path: path of image file'
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

scriptDir=$(dirname $0)

ip=$1
path=$2
img64=$(base64 $path)
echo "size: ${#img64}"
image="\"image\" : \"$img64\""

echo "run curl with POST and image = $path ..."
url="http://${ip}/invocations"
echo "URL: $url"

echo "{$image, \"session_id\" : \"from_test_web_server_curl\"}" | curl -X POST $url -H 'content-type: application/json' -d @-
echo
