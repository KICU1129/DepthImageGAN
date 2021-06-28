FROM pytorch/pytorch
USER root

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN apt-get install -y vim less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

#RUN python -m pip install torch torchvision
RUN python -m pip install matplotlib
RUN python -m pip install seaborn

RUN python -m pip install opencv-python
RUN python -m pip install opencv-contrib-python
RUN apt-get install -y libgl1-mesa-dev

RUN python -m pip install pandas
RUN python -m pip install mlflow







