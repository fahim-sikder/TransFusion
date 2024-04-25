#!/bin/bash

docker run --gpus device=0 -it --rm \
    -v $(pwd)/../:/home/$USER \
    fldgm \
    /bin/bash -c "cd /home/$USER; exec /bin/bash"
