FROM python:3.7-slim-buster

RUN apt -y update && apt install -y --no-install-recommends \
    libgomp1 \
    && apt clean    

RUN pip3 install lightgbm \
                 numpy \
                 optuna \
                 pandas \ 
                 scipy \
                 scikit-learn \

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]