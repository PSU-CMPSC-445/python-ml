#!/usr/bin/env bash

JOB_NAME="lego_classifier_$(date +"%Y%m%d_%H%M%S")"

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --staging-bucket=gs://cmpsc445-staging \
    --job-dir=gs://cmpsc445-models \
    --package-path=trainer \
    --module-name=trainer.task \
    --region=us-east1 \
    --runtime-version=2.4 \
    --python-version=3.7 \
    -- \
    --time-id=${JOB_NAME} \
    --batch-size=32 \
    --num-epochs=32