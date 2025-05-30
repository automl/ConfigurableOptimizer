#!/bin/bash

# Define the base URL and datasets
server=$1

base_url_stanford="http://downloads.cs.stanford.edu/downloads/taskonomy_data"
base_url_epfl="https://datasets.epfl.ch/taskonomy"

if [ "$server" == "stanford" ]; then
    base_url=$base_url_stanford
elif [ "$server" == "epfl" ]; then
    base_url=$base_url_epfl
else
    echo "Error: Invalid server. Please specify 'stanford' or 'epfl'."
    exit 1
fi

datasets=("wainscott" "tolstoy" "klickitat" "pinesdale" "stockman" "beechwood" "coffeen" "corozal" \
"benevolence" "eagan" "forkland" "hanson" "hiteman" "ihlen" "lakeville" "lindenwood" \
"marstons" "merom" "newfields" "pomaria" "shelbyville" "uvalda" "cosmos" "leonardo")


task_types=("rgb" "class_object" "class_scene")


dataset_dir="datasets/taskonomydata_mini"


for dataset in "${datasets[@]}"; do
  
  mkdir -p "$dataset_dir/$dataset"
  
  for task_type in "${task_types[@]}"; do
    file_url="${base_url}/${dataset}_${task_type}.tar"
    file_path="$dataset_dir/$dataset/${dataset}_${task_type}.tar"
    
    if [ -d "$dataset_dir/$dataset/$task_type" ]; then
      echo "$dataset_dir/$dataset/$task_type"
      echo "data already exists"
      continue
    fi
    # Check if the file already exists
    if [ ! -f "$file_path" ]; then
      echo "Downloading $file_url..."
      wget "$file_url" -P "$dataset_dir/$dataset"
    else
      echo "File ${dataset}_${task_type}.tar already exists. Skipping download."
    fi

    echo "Extracting ${dataset}_${task_type}.tar..."
    tar -xvf "$file_path" -C "$dataset_dir/$dataset"
    rm "$file_path"
  done
done
