# this is a google supplied pytorch image
FROM gcr.io/cloud-aiplatform/training/pytorch-gpu.1-4:latest

RUN pip install pandas google-cloud-storage

# this specific image does not need this.
# RUN gcloud auth activate-service-account --key-file=/tmp/keys/gcp.json

# Copy all the trainer code 
WORKDIR /root
COPY trainer/mnist.py ./trainer/mnist.py
COPY trainer/task.py ./trainer/task.py

# bake data into image?
# COPY data/train.csv ./data/train.csv

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "/root/trainer/task.py"]
