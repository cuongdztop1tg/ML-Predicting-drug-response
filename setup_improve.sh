#!/bin/bash --login
# set -e

# Get current dir and model dir
model_path=$PWD
echo "Model path: $model_path"
model_name=$(echo "$model_path" | awk -F '/' '{print $NF}')
echo "Model name: $model_name"

# Clone IMPROVE lib (if needed) and checkout the branch/tag
cd ../
improve_lib_path=$PWD/IMPROVE
# improve_branch="develop"
improve_branch="develop"
if [ -d $improve_lib_path ]; then
    echo "IMPROVE repo exists in ${improve_lib_path}"
else
    git clone https://github.com/JDACS4C-IMPROVE/IMPROVE.git
fi
cd IMPROVE
git checkout $improve_branch
cd ../$model_name

# Download official IMPROVE CSA benchmark data if needed.
data_dir="$PWD/csa_data/raw_data"
if [ ! -d "$data_dir/x_data" ] || [ ! -d "$data_dir/y_data" ] || [ ! -d "$data_dir/splits" ]; then
    echo "Download official IMPROVE CSA benchmark data"
    bash "$improve_lib_path/benchmark_data/drug_response_prediction/scripts/get-benchmarks" "$data_dir"
else
    echo "Official IMPROVE CSA data folder already exists"
fi

# Env vars
export IMPROVE_DATA_DIR="$data_dir"
export PYTHONPATH=$PYTHONPATH:$improve_lib_path

echo
echo "IMPROVE_DATA_DIR: $IMPROVE_DATA_DIR"
echo "PYTHONPATH: $PYTHONPATH"
