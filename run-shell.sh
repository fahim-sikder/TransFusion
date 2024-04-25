#!/bin/bash

docker run --gpus device=0 -it --rm \
    -v $(pwd)/../:/home/$USER \
    transfusion \
    /bin/bash -c "cd /home/$USER; exec /bin/bash"
