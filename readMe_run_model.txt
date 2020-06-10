In order to reproduce the experiment results, please follow the instructions below to run source code. Here we acknowledge some users who open repositories on Github for us to reuse some of their code to do experiments

1. Before running the models, please first look at "data_download.txt" to download public data
2. Please install some packages in "requirements.txt"


## [1] Disentangling experiments
1)>> run visdom server: bash run_server.sh
2)>> bash run_dsprites_pid_c18.sh

## Image Generation experiments
1) bash run_server.sh
2) bash run_celeba_PID_z500_KL200_d128.sh


## Language Modeling
1) for text generation on PTB data, please run
>> bash run_vae_transform_ptb_pid.sh


2) for Dialog generation on SW data:
>> unzip data.zip to get the SW data
>> bash run_pid.sh
