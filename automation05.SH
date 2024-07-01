#!/bin/bash

# Define the log files
training_log="training.log"
evaluation_log="evaluation.log"

# Create the log files (truncate if they already exist)
# shellcheck disable=SC2188
> "$training_log"
# shellcheck disable=SC2188
> "$evaluation_log"

# Get the current working directory and assign it to EXECUTION_PATH
EXECUTION_PATH=$(pwd)

# Print the execution path
echo "execution path: $EXECUTION_PATH"

# Print the separator
echo "***************************************************"
echo "***************************************************"
echo "                  TRAINING PHASE"
echo "***************************************************"
echo "***************************************************"

#source /path/to/your/venv/bin/activate
echo "activating virtual environment"

echo "Starting TRAINING"
echo "---------------------------------------------------"
echo "               TRAINING model01-plain "
echo "---------------------------------------------------"
python3 ./model01-plain/training.py | tee -a "$training_log"

echo "---------------------------------------------------"
echo "              TRAINING model02-fourO "
echo "---------------------------------------------------"
python3 ./model02-fourO/training.py | tee -a "$training_log"

echo "---------------------------------------------------"
echo "        TRAINING model03-ConvBiLSTMclassifier "
echo "---------------------------------------------------"
python3 ./model03-ConvBiLSTMclassifier/training.py | tee -a "$training_log"

echo "---------------------------------------------------"
echo "             TRAINING model04-fiveO "
echo "---------------------------------------------------"
python3 ./model04-fiveO/training.py | tee -a "$training_log"


echo "***************************************************"
echo "***************************************************"
echo "                EVALUATION PHASE"
echo "                   PerSpacer"
echo "***************************************************"
echo "***************************************************"


echo "---------------------------------------------------"
echo "               EVALUATING model01-plain "
echo "---------------------------------------------------"
python3 ./perspacer_evaluation.py --model_dir ./model01-plain/Model/model/ --dataset_dir ./_data/datasets/batched/bert-base-multilingual-uncased/ | tee -a "$evaluation_log"

echo "---------------------------------------------------"
echo "              EVALUATING model02-fourO "
echo "---------------------------------------------------"
python3 ./perspacer_evaluation.py --model_dir ./model02-fourO/Model/model/ --dataset_dir ./_data/datasets/batched/bert-base-multilingual-uncased/ | tee -a "$evaluation_log"

echo "---------------------------------------------------"
echo "        EVALUATING model03-ConvBiLSTMclassifier "
echo "---------------------------------------------------"
python3 ./perspacer_evaluation.py --model_dir ./model03-ConvBiLSTMclassifier/Model/model/ --dataset_dir ./_data/datasets/batched/bert-base-multilingual-uncased/ | tee -a "$evaluation_log"

echo "---------------------------------------------------"
echo "             EVALUATING model04-fiveO "
echo "---------------------------------------------------"
python3 ./perspacer_evaluation.py --model_dir ./model04-fiveO/Model/model/ --dataset_dir ./_data/datasets/batched/bert-base-multilingual-uncased/ | tee -a "$evaluation_log"


echo "***************************************************"
echo "***************************************************"
echo "                EVALUATION PHASE"
echo "                   PerSpaCor"
echo "***************************************************"
echo "***************************************************"


echo "---------------------------------------------------"
echo "               EVALUATING model01-plain "
echo "---------------------------------------------------"
python3 ./perspacor_evaluation.py --alpha 0.5 --model_dir ./model01-plain/Model/model/ --dataset_dir ./_data/datasets/batched/bert-base-multilingual-uncased/ --injected_dir ./_data/datasets/batched/bert-base-multilingual-uncased/injected/ | tee -a "$evaluation_log"

echo "---------------------------------------------------"
echo "              EVALUATING model02-fourO "
echo "---------------------------------------------------"
python3 ./perspacor_evaluation.py --alpha 0.5 --model_dir ./model02-fourO/Model/model/ --dataset_dir ./_data/datasets/batched/bert-base-multilingual-uncased/ | tee -a "$evaluation_log"

echo "---------------------------------------------------"
echo "        EVALUATING model03-ConvBiLSTMclassifier "
echo "---------------------------------------------------"
python3 ./perspacor_evaluation.py --alpha 0.5 --model_dir ./model03-ConvBiLSTMclassifier/Model/model/ --dataset_dir ./_data/datasets/batched/bert-base-multilingual-uncased/ | tee -a "$evaluation_log"

echo "---------------------------------------------------"
echo "             EVALUATING model04-fiveO "
echo "---------------------------------------------------"
python3 ./perspacor_evaluation.py --alpha 0.5 --model_dir ./model04-fiveO/Model/model/ --dataset_dir ./_data/datasets/batched/bert-base-multilingual-uncased/ | tee -a "$evaluation_log"
