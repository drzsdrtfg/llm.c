# Cuda model training (cudadev; >12.4)
```bash
git clone https://github.com/drzsdrtfg/llm.c.git
cd llm.c
pip install tqdm tiktoken requests datasets transformers
# for me, CUDA 12 (run `nvcc --version`) running on Linux x86_64 Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcudnn9-dev-cuda-12
# "install" cudnn-frontend to ~/
git clone https://github.com/NVIDIA/cudnn-frontend.git

# install MPI (optional, if you intend to use multiple GPUs)
sudo apt install openmpi-bin openmpi-doc libopenmpi-dev
# tokenize the FineWeb dataset 10B tokens sample (takes ~1 hour, get lunch?)
# writes ~19GB of raw GPT-2 tokens to dev/data/fineweb10B
# and ~46GB in ~/.cache/huggingface/datasets/HuggingFaceFW___fineweb
python dev/data/fineweb.py --version 10B
# compile llm.c (mixed precision, with cuDNN flash-attention)
# first compilation is ~1 minute, mostly due to cuDNN
make train_gpt2cu USE_CUDNN=1

# train on a single GPU
./train_gpt2cu \
    -i "dev/data/fineweb10B/fineweb_train_*.bin" \
    -j "dev/data/fineweb10B/fineweb_val_*.bin" \
    -o log124M \
    -e "d12" \
    -b 64 -t 1024 \
    -d 524288 \
    -r 1 \
    -z 1 \
    -c 0.1 \
    -l 0.0006 \
    -q 0.0 \
    -u 700 \
    -n 5000 \
    -v 250 -s 20000 \
    -h 1

# if you have multiple GPUs (e.g. 8), simply prepend the mpi command, e.g.:
# mpirun -np 8 ./train_gpt2cu \ ... (the rest of the args are same)
```
## VSC problems with corresponding solutions

### Independent terminals
```bash
exec bash --login
touch ~/.no_auto_tmux
#>Disconnect and reconnect>solved?
```
---

<details>
<summary>📌 Note</summary>
Ensure you have the necessary hardware and CUDA setup for GPU training. Adjust the number of GPUs in the multi-GPU command as needed.
</details>

