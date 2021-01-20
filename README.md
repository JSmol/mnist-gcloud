# Minimal torch on GCP with GPU example

Using `sh cloud.sh` will:
- Build the docker image with nvidia support \(the base image is supplied by google\).
- Upload it to google cloud platform.
- Start a job that can be found in AI-Platform > Jobs.
- Stream the output logs to the terminal session.

The jobs will:
- Take several arguments from command line.
- Run on the GPU if it is avaible.
- Define a sequential CNN.
- Train the model on the dataset \(potentially stored on GCP\) downloaded from [Kaggle](https://www.kaggle.com/c/digit-recognizer).
- Upload the trained model parameters and the evaluation results to google storage bucket.

The scripts are easiliy modified to add features such as reading a pre trained models from google storage to resume a former training session. Ideally specific paths should be specified 

*This repo is an example for future projects that I cannot reasonably run locally. I followed the google supplied examples closesly, but this is smaller baseline suited to my own one-man-army projects.*
