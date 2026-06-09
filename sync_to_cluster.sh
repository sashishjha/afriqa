#!/bin/bash

# Sync local changes to the remote Linux cluster (RBCCPS, IISc)
# Excludes local virtual environment, python caches, outputs, and git history.

rsync -avz --exclude='.venv' \
           --exclude='__pycache__' \
           --exclude='outputs/' \
           --exclude='.git/' \
  /Users/sashishjha/Downloads/afriqa/ \
  sashishj@10.72.30.28:~/projects/afriqa/
