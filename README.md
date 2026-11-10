# Mutually Distilled Multi-View RGB-Event Recognition

This repository contains the reproducible implementation of the three-branch
RGB-Event training and dynamic hierarchical model merging pipeline.

## Configuration

1. `bash train.sh` trains the frame, stream, and voxel branches with
   classification, logit-level mutual distillation, feature-level mutual
   distillation, and relation preservation.
2. `bash merge.sh` merges historical full checkpoints with frequency-aware
   branch experts. Set the checkpoint variables before running it.

3. `CHECKPOINT=... bash test.sh` evaluates the merged checkpoint with fused
   logits.

Before training, set these environment variables for the dataset.
