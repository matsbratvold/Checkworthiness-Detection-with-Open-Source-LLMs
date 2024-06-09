# Check-Worthiness Detection With Open Source LLMs

This repository contains code used for check-worthiness detection with Open Source Large Language Models.
The work is conducted as part of a Master's thesis in MTDT Computer Science at the Norwegian University of School and Technology (NTNU) during the spring semester of 2024.

## Repository structure

### [Claimbuster-spotter-master folder](/claimbuster-spotter-master)

Contains the baseline models from [Meng et al. 2020](http://arxiv.org/abs/2002.07725). It contains only slight modifications from their [Github repository](https://github.com/utaresearch/claimbuster-spotter).

### [Data folder](/data)

Contains the datasets used during the conducted experiments. This includes the ClaimBuster and CheckThat 2021 Task 1a datasets used for check-worthiness detection and the LIAR and RAWFC datasets used for factual verification.

### [Figure folder](/figures)

Contains plots that explore properties of the aforementioned datasets.

### [Src folder](/src)

Contains source code used for exploring the data and performing experiments.
*llm.py* is used to perform inference with Open Source LLMs in zero-shot, few-shot and CoT settings while *lora_finetuning.py* is used to perform LoRA fine-tuning.
*experiments.ipynb* is used to run the actual experiments while the other jupyter notebooks are used to explore the different datasets.

### [Results folder](/results)

Contains experimental results. For the first four experiments, it is structured into folders in the following way: *{{dataset}}/{{model}}/{{promptType}}/{{ICLUsage}}*
where *dataset* is either ClaimBuster or CheckThat, *model* is either Mistral Instruct, Mixtral Instruct or LLama 2 Chat, *promptType* is standard, CoT or LoRA and *ICLUsage* is zero-shot or few-shot.
Each configuration contains check-worthiness predictions, cross validation results and a confusion matrix.

For LIAR and RAWFC, it contains check-worthiness predictions and groups claims by check-worthiness and truthfulness labels.

## Running the experiments

In order to run the experiments, it is most beneficial to use Anaconda (Conda command line tool) in order to reproduce the environments that was used during the Master's thesis.
Note that this is only tested on Linux, so I would recommend using WSL if trying to reproduce the results on Windows. If on MAC, you will probably have to install the required packages yourself.
See the [Conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) for more details about installation and how to use conda environments.

### LLMs

After installing conda, you can reproduce the environment used during LLM inference by running the command ````conda create -f environment.yml```` within the root folder of the repository.
This should create an environment called *CWD* that should be able to run the *experiment.ipynb* jupyter notebook within the *src* folder. Note that to properly run the LLMs, it will require a sufficiently large LLM, preferably at least 32 GB to run the largest Mixtral Instruct model. You also need to make sure that GPU support is properly set up in with Tensorflow. See the [Tensorflow documentation](https://www.tensorflow.org/install/pip) for more information.

### Baseline models

In order to run the baseline models, you would need to use a different environment that is located in the *claimbuster-spotter-master* folder. Simply change to this directory and run the same conda command as before to create an environment called *ClaimBuster*.
If you experience errors with the wrong version of *numpy* being installed, you should unistall the package using ````pip uninstall numpy```` and then reinstalling with ````pip install numpy==1.18.5````
Note that this relies on Python 3.6 and an old version of Tensorflow and CUDA in order to be compatible with the original code from Meng et al. This could lead to trouble with some GPUs such as Nvidia A100. Preferably, the code should have been updated to support the newest versions of TensorFlow and CUDA, but there was not enough time to do this during the Master's thesis.

Using this conda environment, you should be able to run the different jupyter notebooks within the *claimbuster-spotter-master* folder. There is one notebook for the SVM baseline, one for the BiLSTM baseline and one for the adversarial transformer.
