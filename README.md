# Block Insertion using Robo Simulation

GitHub Repository of this project can be found from below link:

[GitHub Repository](https://github.com/karankumbhar47/Robotic-Simulation)


**Abstract:** In our study, we introduce an approach aimed at improving collaboration among decentralized multi-agent robots engaged in physical tasks.In this project each agent's intention is shared with others and translated into an overhead 2D map aligned with visual observations. This idea is complemented by the spatial action maps framework, aligning state and action representations spatially. This alignment introduces inductive biases that encourage cooperative behaviors like object passing and collision avoidance. Through experiments in various multi-agent environments featuring heterogeneous robot teams, our results suggest that incorporating spatial intention maps has the potential to improve performance in mobile manipulation tasks and enhance cooperative behaviors. While these findings are promising, we acknowledge the ongoing nature of research in this domain and look forward to further exploration and refinement.

## Installation

We Have used a [`conda`](https://docs.conda.io/en/latest/miniconda.html) environment for this codebase. The following commands will set up a new conda environment with the necessary requirements (tested on Ubuntu 22.04.3 LTS):

```bash
# Creating and activating new conda env
conda create -y -n my-conda-env python=3.7.10
conda activate my-conda-env

# Installing mkl numpy
conda install -y numpy==1.19.2

# Installing pytorch
conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

# Installing pip requirements
pip install -r requirements.txt

# Installing shortest paths module
cd shortest_paths
python setup.py build_ext --inplace
```

## Quickstart

We provide pre-trained policies for each test environment. To obtain these pre-trained policies, use the `download-pretrained.sh` script. Running this script will fetch the pre-trained policies and store their configurations in the `logs` directory, and their network weights in the `checkpoints` directory. Execute the following command to run the script:


```bash
./download-pretrained.sh
```

Then you can use `demos.py` to run a pretrained policy in the simulation environment. Here are a few examples you can try:

```bash
# 4 lifting robots
python demos.py --config-path logs/20201217T171233203789-lifting_4-small_divider-ours/config.yml
python demos.py --config-path logs/20201214T092812731965-lifting_4-large_empty-ours/config.yml

# 4 pushing robots
python demos.py --config-path logs/20201214T092814688334-pushing_4-small_divider-ours/config.yml
python demos.py --config-path logs/20201217T171253620771-pushing_4-large_empty-ours/config.yml

# 2 lifting + 2 pushing
python demos.py --config-path logs/20201214T092812868257-lifting_2_pushing_2-large_empty-ours/config.yml
```

You can see the pretrained policy running in the PyBullet GUI that pops up. Here are a few examples of what it looks like (4x speed):

![](https://user-images.githubusercontent.com/6546428/111895630-3842bc80-89d1-11eb-9150-1364f80e3a26.gif) | ![](https://user-images.githubusercontent.com/6546428/111895627-35e06280-89d1-11eb-9cf7-0de0595ae68f.gif) 
:---: | :---: 
`lifting_4-small_divider` | `lifting_2_pushing_2-large_empty` 

You can try running `demos.py` without specifying a config path, and it will list all policies in the `logs` directory and allow you to pick one to run:

```bash
python demos.py
```

---


## Training in the Simulation Environment

In the [`config/experiments`](config/experiments) directory, you'll find template config files utilized in all experiments. To initiate a training session, supply one of these template config files to the `train.py` script. For instance, the command below will initiate the training of a policy in the `SmallDivider` environment:


```bash
python train.py --config-path config/experiments/ours/lifting_4-small_divider-ours.yml
```

Upon running the training script, it will generate a log directory and a checkpoint directory for the new training session within `logs/` and `checkpoints/`, respectively. Within the log directory, a new configuration file named `config.yml` will be created. This file houses the configuration variables for the training run and can be utilized to either resume training or load a trained policy for evaluation.


### Conclusion

In conclusion, our work on block insertion using reinforcement learning offers a new perspective on enhancing collaboration among decentralized multi-agent robots in physical tasks. The introduction of spatial intention maps and their alignment with visual observations, coupled with the spatial action maps framework, demonstrates promising results in improving coordination and encouraging cooperative behaviors.
