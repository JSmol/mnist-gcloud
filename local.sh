IMAGE_URI=torch-gpu:mnist
docker build -t ${IMAGE_URI} .
docker run \
  -v $GOOGLE_APPLICATION_CREDENTIALS:/tmp/keys/gcp.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/gcp.json \
  ${IMAGE_URI} \
  --data-path=gs://mnist-train \
  --bucket-name=python-trained-models \
  --epochs=1 \
  --batch-size=100 \
  --learning-rate=0.0001 \
  --weight-decay=0