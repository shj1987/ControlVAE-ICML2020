# ControlVAE-ICML2020
We release the source code for our paper "ControlVAE: Controllable Variational Autoencoder" published at ICML 2020. It can be used for disentanglement representation learning, text generation and image generation.

If you use our source code, please cite our paper: <br />
@article{shao2020controlvae, <br />
  title={ControlVAE: Controllable Variational Autoencoder}, <br />
  author={Shao, Huajie and Yao, Shuochao and Sun, Dachun and Zhang, Aston and Liu, Shengzhong and Liu, Dongxin and Wang, Jun and Abdelzaher, Tarek}, <br />
  journal={Proceedings of the 37th International Conference on Machine Learning (ICML)}, <br />
  year={2020},<br />
  address={Vienna, Austria}<br />
}


---
### :bulb: Recent Changes :bulb:
#### 2020.06.12
* Updating the language model of text generation on PTB data. Previously, I forgot to commit it from my AWS server. Please clone the latest version. If you have any questions, please feel free to contact me (hshao5@illinois.edu).

---
### Download the public data
#### -Dsprites for Disentangling application
1. Dsprite Data for disentanglement
  * Step1: Enter the path "./Disentangling/"
  * Step2: sh prepare_data.sh dsprites

#### -CelebA data
2. CelebA data for Image Generation Application
  * Step 1: Download public data in the website: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
  * Step2: first download img_align_celeba.zip and put in data directory like below
      ./data/img_align_celeba.zip
  * Step3: Enter the folder "Image_generation", and then run scrip file
  ```
      $ bash prepare_data.sh CelebA
      $ python3 data_split.py // split the data into training and testing data
  ```

#### -PTB data
3. PTB data for Language modeling Application
  ```
  * Step1: enter the path "./Language_modeling/Text_gen_PTB"
  * Step2: then run (install Texar torch first) python prepare_data.py --data ptb
  ```


#### -Switchboard(SW) data
4. Switchboard(SW) data for Dialog-generation
 * Please Download Glove word embeddings from http://nlp.stanford.edu/data/glove.twitter.27B.zip. The default setting use 200 dimension word embedding trained on Twitter.
Then unzip and save the data into the path "./glove/glove.twitter.27B.200d.txt"

 * We already download the SW data and zip it in the path "./Language_modeling/NeuralDialog/data.

---
### Before running the source code, please read requirements.txt to install the dependencies

---
### In order to reproduce the experiment results, please follow the instructions below to run source code. 

  1. Before running the models, please first look at "data_download.txt" to download public data
  2. Please install some packages in "requirements.txt"


#### [1] Disentangling experiments
  * run visdom server: $ bash run_server.sh
  * bash run_dsprites_pid_c18.sh

#### [2] Image Generation experiments
  * bash run_server.sh
  * bash run_celeba_PID_z500_KL200_d128.sh


#### [3] Language Modeling
 * for text generation on PTB data, please run: bash run_vae_transform_ptb_pid.sh

 * for Dialog generation on SW data, please run: $ bash run_pid.sh

---
### :heart: We thank the following users who open repositories on Github for us to build on for our experiments
 * Texar-pytorch https://github.com/asyml/texar-pytorch
 * 1Konny (Beta-VAE) https://github.com/1Konny/Beta-VAE
 * snakeztc (NeuralDialog-CVAE) https://github.com/snakeztc/NeuralDialog-CVAE

