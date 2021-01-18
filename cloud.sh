# CHANGE the URI to match project
IMAGE_URI=gcr.io/aesthetic-fx-300721/torch-gpu:mnist
docker build -t ${IMAGE_URI} .

# CHANGE the JOB_NAME to match project
JOB_NAME=MNIST_$(date +%Y_%m_%d_%s)
REGION=us-central1

docker build -t ${IMAGE_URI}
docker push ${IMAGE_URI}

# this sends the job to ai platform jobs
# after -- \ the arguments will be passed to the docker image!
# NOTE the image can access gs://
gcloud beta ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --scale-tier BASIC_GPU \
    -- \
    --data-path=gs://mnist-train \
    --epochs=100 \
    --batch-size=100 \
    --learning-rate=0.001 \
    --weight-decay=0.001 \
    --beta1=0.5 \
    --beta2=0.9

# # Stream the logs from the job
gcloud ai-platform jobs stream-logs ${JOB_NAME}

# this is for ai platform (unified) which doesnt seem to do GPU yet...
# gcloud beta ai custom-jobs create \
#   --region=$REGION \
#   --display-name=$JOB_NAME \
#   --config=config.yaml
