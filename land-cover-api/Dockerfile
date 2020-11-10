FROM mcr.microsoft.com/aiforearth/blob-python:1.0

RUN apt-get update && apt-get install -y libglib2.0-0 libspatialindex-dev
RUN apt install libgl1-mesa-glx -y
RUN apt install -y unzip libgl1

# Note: supervisor.conf reflects the location and name of your api code.
# If the default (./my_api/runserver.py) is renamed, you must change supervisor.conf
# Copy your API code
COPY ./api /app/api/
COPY ./supervisord.conf /etc/supervisord.conf

# startup.sh is a helper script
COPY ./startup.sh /
RUN chmod +x /startup.sh

RUN /usr/local/envs/ai4e_py_api/bin/pip install -r /app/api/requirements.txt

RUN echo "source activate ai4e_py_api" >> ~/.bashrc 

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

ENV API_PREFIX=/v2

# Expose the port that is to be used when calling your API
EXPOSE 80
HEALTHCHECK --interval=1m --timeout=3s --start-period=20s \
CMD curl -f http://localhost/ || exit 1
ENTRYPOINT [ "/startup.sh" ]
