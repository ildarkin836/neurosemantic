FROM python:3.10-slim

WORKDIR /opt/neurosemantic
RUN python3 -m pip install --no-cache-dir nvidia-pyindex==1.0.9
COPY ./requirements.txt /opt/neurosemantic/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /opt/neurosemantic/requirements.txt
COPY ./ /opt/neurosemantic


