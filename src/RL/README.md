Install CUDA for WSL - https://docs.nvidia.com/cuda/wsl-user-guide/index.html
pip install stable-baselines3[extra]
- installs gym, pytorch for you
pip install gym[atari]
pip install autorom[accept-rom-license]
pip install sb3-contrib
- Installs the recurrent version of PPO


Run src/RL/test/torch_test.py to verify torch can see your Cuda enabled device.
Then it should automatically use the GPU when running PPO or recurrent PPO. 