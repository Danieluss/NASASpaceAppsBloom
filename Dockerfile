FROM tensorflow/tensorflow:1.14.0-gpu-py3

ENV APP_DIR=/opt/bloom

RUN apt-get install -y python3-pip

RUN mkdir -p $APP_DIR

COPY . $APP_DIR

RUN cd $APP_DIR \
    && /bin/bash install.sh

ENTRYPOINT /bin/bash

CMD ["python3", "visualise.py"]