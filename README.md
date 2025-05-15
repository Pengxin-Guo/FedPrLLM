# FedPrLLM
Exploring Federated Pruning for Large Language Models

## Installation

Our code is based on Python version 3.10 and PyTorch version 2.1.0. 
You can install all the dependencies with the following command:
```shell
conda create -n fedprllm python=3.10
conda activate fedprllm
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.36.2 datasets==2.16.1 wandb sentencepiece==0.1.99 accelerate==0.26.1
```

## Pruning

Now, we can prune a LLM with FedPrLLM:

```shell
python main.py --model huggyllama/llama-7b --sparsity_ratio 0.5 --method fedprllm --comparison_group layer --save out/llama-1_7b/sparity_0.5/
```

## Acknowledgement

We would like to thank the authors for releasing the public repository: [Wanda](https://github.com/locuslab/wanda).