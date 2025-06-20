# Reinforcement Learning DQN Algorithm implementation

This is DQN Algorithm from scratch for educational purposes. Use on your own risk.

## Setup

```sh
conda create -n rl_dqn python=3.12 -y
conda activate rl_dqn

python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -m pip install -r requirements.txt
python -m pip install .
```

## Train

```sh
dqn-train -c cfg/space_invaders3.yaml
```
