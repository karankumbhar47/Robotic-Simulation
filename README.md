# Block Insertion using RL

GitHub Repository of this project can be found from below link:

[GitHub Repository](https://github.com/karankumbhar47/Robotic-Simulation)


**Abstract:** In our study, we introduce an approach aimed at improving collaboration among decentralized multi-agent robots engaged in physical tasks. We propose spatial intention maps, a new representation for multi-agent vision-based deep reinforcement learning, with the goal of enhancing coordination among decentralized mobile manipulators. In this representation, each agent's intention is shared with others and translated into an overhead 2D map aligned with visual observations. This idea is complemented by the spatial action maps framework, aligning state and action representations spatially. This alignment introduces inductive biases that encourage cooperative behaviors like object passing and collision avoidance. Through experiments in various multi-agent environments featuring heterogeneous robot teams, our results suggest that incorporating spatial intention maps has the potential to improve performance in mobile manipulation tasks and enhance cooperative behaviors. While these findings are promising, we acknowledge the ongoing nature of research in this domain and look forward to further exploration and refinement.

## Installation

We recommend using a [`conda`](https://docs.conda.io/en/latest/miniconda.html) environment for this codebase. The following commands will set up a new conda environment with the correct requirements (tested on Ubuntu 22.04.3 LTS):

```bash
# Create and activate new conda env
conda create -y -n my-conda-env python=3.7.10
conda activate my-conda-env

# Install mkl numpy
conda install -y numpy==1.19.2

# Install pytorch
conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

# Install pip requirements
pip install -r requirements.txt

# Install shortest paths module (used in simulation environment)
cd shortest_paths
python setup.py build_ext --inplace
```

## Quickstart

We provide pretrained policies for each test environment. The `download-pretrained.sh` script will download the pretrained policies and save their configs and network weights into the `logs` and `checkpoints` directories, respectively. Use the following command to run it:

```bash
./download-pretrained.sh
```

You can then use `demos.py` to run a pretrained policy in the simulation environment. Here are a few examples you can try:

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

You should see the pretrained policy running in the PyBullet GUI that pops up. Here are a few examples of what it looks like (4x speed):

![](https://user-images.githubusercontent.com/6546428/111895630-3842bc80-89d1-11eb-9150-1364f80e3a26.gif) | ![](https://user-images.githubusercontent.com/6546428/111895627-35e06280-89d1-11eb-9cf7-0de0595ae68f.gif) 
:---: | :---: 
`lifting_4-small_divider` | `lifting_2_pushing_2-large_empty` | `rescue_4-small_empty`

You can also run `demos.py` without specifying a config path, and it will list all policies in the `logs` directory and allow you to pick one to run:

```bash
python demos.py
```

---

While the focus of this work is on multi-agent, the code also supports single-agent training. We provide a few pretrained single-agent policies which can be downloaded with the following command:

```bash
./download-pretrained.sh --single-agent
```

Here are a few example pretrained single-agent policies you can try:

```bash
# 1 lifting robot
python demos.py --config-path logs/20201217T171254022070-lifting_1-small_empty-base/config.yml

# 1 pushing robot
python demos.py --config-path logs/20201214T092813073846-pushing_1-small_empty-base/config.yml
```

Here is what those policies look like when running in the PyBullet GUI (2x speed):

![](https://user-images.githubusercontent.com/6546428/111895625-34169f00-89d1-11eb-8687-689122e6b3f2.gif) | ![](https://user-images.githubusercontent.com/6546428/111895631-38db5300-89d1-11eb-9ad4-81be3908f383.gif)
:---: | :---:
`lifting_1-small_empty` | `pushing_1-small_empty` | `rescue_1-small_empty`

## Training in the Simulation Environment

The [`config/experiments`](config/experiments) directory contains the template config files used for all experiments. To start a training run, you can provide one of the template config files to the `train.py` script. For example, the following will train a policy on the `SmallDivider` environment:

```bash
python train.py --config-path config/experiments/ours/lifting_4-small_divider-ours.yml
```

The training script will create a log directory and checkpoint directory for the new training run inside `logs/` and `checkpoints/`, respectively. Inside the log directory, it will also create a new config file called `config.yml`, which stores training run config variables and can be used to resume training or to load a trained policy for evaluation.


### Simulation Environment

To interactively explore the simulation environment using our dense action space (spatial action maps), you can use `tools_simple_gui.py`, which will load an environment and allow you to click on the agent's local overhead map to select navigational endpoints (each pixel is an action). Some robot types (such as lifting) have a 2-channel action space, in which case you would use left click to move, and right click to move and then attempt an end effector action at the destination (such as lift or throw).

```bash
python tools_simple_gui.py
```

Note that `tools_simple_gui.py` currently only supports single-agent environments. For multi-agent environments, you can use `tools_interactive_gui.py` which has many more features, including control of multiple agents:

```bash
python tools_interactive_gui.py
```
### Conclusion

In conclusion, our work on block insertion using reinforcement learning offers a new perspective on enhancing collaboration among decentralized multi-agent robots in physical tasks. The introduction of spatial intention maps and their alignment with visual observations, coupled with the spatial action maps framework, demonstrates promising results in improving coordination and encouraging cooperative behaviors.