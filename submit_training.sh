#!/usr/bin/env bash

gcloud ai-platform jobs submit training lego_classifier_15 \
    --staging-bucket=gs://cmpsc445-staging \
    --job-dir=gs://cmpsc445-models \
    --package-path=trainer \
    --module-name=trainer.task \
    --region=us-east1 \
    --runtime-version=2.4 \
    --python-version=3.7 \
    -- \
    --image-height=400 \
    --image-width=400 \
    --batch-size=32 \
    --num-epochs=1