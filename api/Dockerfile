FROM python:3.7
# make directories
RUN mkdir -p /home/project/app
RUN mkdir -p /home/project/app/model
RUN mkdir -p /home/project/app/dict
WORKDIR /home/project/app
# copy and install packages for flask
COPY requirements.txt /home/project/app
RUN pip install --no-cache-dir -r ./requirements.txt
# copy contents from local to the docker container
COPY . /home/project/app
COPY ./lib/model /home/project/app/model
COPY ./lib/data/labelDict.pickle /home/project/app/dict/labelDict.pickle