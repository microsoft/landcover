FROM mcr.microsoft.com/aiforearth/blob-python:latest

# Copy blob connection information
COPY ./blob_mount.json /app/fuse/blob_mount.json

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-glx

RUN /usr/local/envs/ai4e_py_api/bin/pip install flask-cors

RUN /usr/local/envs/ai4e_py_api/bin/pip install rasterio==1.0.12

# Example of installing additonal Anaconda packages - numpy and pandas
RUN echo "source activate ai4e_py_api" >> ~/.bashrc \
    && conda update conda \
    && conda install -c conda-forge -n ai4e_py_api numpy pandas scipy \
    && conda install -c conda-forge gdal=2.2.2 \
    #&& conda install -c conda-forge -n ai4e_py_api rasterio \
    && conda install -c conda-forge -n ai4e_py_api fiona shapely pyproj rtree \
    && conda install -c conda-forge -n ai4e_py_api opencv 

# Note: supervisor.conf reflects the location and name of your api code.
# If the default (./my_api/runserver.py) is renamed, you must change supervisor.conf
# Copy your API code
COPY ./api /app/api/
COPY ./supervisord.conf /etc/supervisord.conf
# startup.sh is a helper script
COPY ./startup.sh /
RUN chmod +x /startup.sh

COPY ./LocalForwarder.config /lf/

# Application Insights keys and trace configuration
ENV APPINSIGHTS_INSTRUMENTATIONKEY= \
    APPINSIGHTS_LIVEMETRICSSTREAMAUTHENTICATIONAPIKEY= \
    LOCALAPPDATA=/app_insights_data \
    OCAGENT_TRACE_EXPORTER_ENDPOINT=localhost:55678

# The following variables will allow you to filter logs in AppInsights
ENV SERVICE_OWNER=AI4E_Test \
    SERVICE_CLUSTER=Local\ Docker \
    SERVICE_MODEL_NAME=base-py\ example \
    SERVICE_MODEL_FRAMEWORK=Python \
    SERVICE_MODEL_FRAMEOWRK_VERSION=3.6.6 \
    ENVSERVICE_MODEL_VERSION=1.0

ENV API_PREFIX=/v1/api

# Expose the port that is to be used when calling your API
EXPOSE 80
HEALTHCHECK --interval=1m --timeout=3s --start-period=20s \
  CMD curl -f http://localhost/ || exit 1
ENTRYPOINT [ "/startup.sh" ]
