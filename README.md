# deepseek-r1
Deploying DeepSeek R1 model on OCI GPU NVIDIA A100 80Gb/H100 80GB with OKE
Running DeepSeek R1 and Qwen2.5 full models on Oracle Cloud Infrastructure (OCI) using A100 80GB and H100 80GB bare metal nodes with 8 GPUs each is an excellent choice for high-performance AI workloads. This article will guide you through the process of setting up and running these models on OCI’s powerful GPU infrastructure.

Setting Up the Environment

First, you’ll need to provision the appropriate bare metal instances on OCI. For this setup, we’ll use:
	1.	BM.GPU.A100-8.80GB - 8x NVIDIA A100 GPUs with 80GB memory each
	2.	BM.GPU.H100-8.80GB - 8x NVIDIA H100 GPUs with 80GB memory each

These instances provide the necessary computational power and memory to run the full versions of DeepSeek R1 and Qwen2.5 models efficiently.

When running large-scale AI workloads on OCI, network latency and bandwidth can impact multi-GPU performance, especially when scaling across multiple nodes. Consider the following:
1.	Choose an OCI region with the closest proximity to your organization or target user base to reduce latency.
2.	Use OCI’s high-bandwidth options such as RDMA clusters (where available) for training or inference across multiple nodes.
3.	Configure VLANs or subnets with sufficient bandwidth to handle the multi-GPU traffic (ideally with a 100 Gbps network backbone where possible).


Installing Required Software

Once your instances are up and running, you’ll need to install the necessary software. OCI provides pre-configured NVIDIA GPU-enabled Data Science images that come with many required packages pre-installed[37]. However, you may need to update or install additional libraries.

BASH
# UPDATE AND INSTALL REQUIRED PACKAGES
SUDO APT-GET UPDATE
SUDO APT-GET INSTALL -Y PYTHON3-PIP

# INSTALL PYTORCH AND OTHER NECESSARY LIBRARIES
PIP3 INSTALL TORCH TORCHVISION TORCHAUDIO
PIP3 INSTALL TRANSFORMERS ACCELERATE BITSANDBYTES

Deploying DeepSeek R1

DeepSeek R1 is a powerful language model that can be easily deployed using Ollama. Here’s how to set it up:

BASH
# INSTALL OLLAMA
CURL -FSSL HTTPS://OLLAMA.COM/INSTALL.SH | SH

# PULL AND RUN DEEPSEEK R1 70B MODEL
OLLAMA RUN DEEPSEEK-R1:70B

This will download and run the full 70B parameter version of DeepSeek R1[5][36].

Deploying Qwen2.5

For Qwen2.5, we’ll use the Hugging Face Transformers library to load and run the model. Here’s a Python script to set up and run Qwen2.5-72B:

PYTHON
IMPORT TORCH
IMPORT TRANSFORMERS

DEF SETUP_QWEN2_5(MODEL_NAME='QWEN/QWEN2-72B-INSTRUCT', DEVICE='CUDA'):
    # LOAD MODEL CONFIGURATION
    CONFIG = TRANSFORMERS.AUTOCONFIG.FROM_PRETRAINED(MODEL_NAME)
    
    # INITIALIZE MODEL WITH 4-BIT QUANTIZATION FOR EFFICIENT LOADING
    MODEL = TRANSFORMERS.AUTOMODELFORCAUSALLM.FROM_PRETRAINED(
        MODEL_NAME, 
        DEVICE_MAP='AUTO',
        LOAD_IN_4BIT=TRUE,
        QUANTIZATION_CONFIG=TRANSFORMERS.BITSANDBYTESCONFIG(
            LOAD_IN_4BIT=TRUE,
            BNB_4BIT_COMPUTE_DTYPE=TORCH.FLOAT16
        ),
        TORCH_DTYPE=TORCH.FLOAT16
    )
    
    # LOAD TOKENIZER
    TOKENIZER = TRANSFORMERS.AUTOTOKENIZER.FROM_PRETRAINED(MODEL_NAME)
    
    RETURN MODEL, TOKENIZER

DEF MULTI_GPU_DEPLOYMENT(MODEL, NUM_GPUS=8):
    IF TORCH.CUDA.DEVICE_COUNT() > 1:
        MODEL = TORCH.NN.DATAPARALLEL(MODEL)
    RETURN MODEL

# LOAD AND DEPLOY QWEN2.5
QWEN_MODEL, QWEN_TOKENIZER = SETUP_QWEN2_5()
QWEN_MODEL = MULTI_GPU_DEPLOYMENT(QWEN_MODEL)

PRINT("QWEN2.5 MODEL LOADED SUCCESSFULLY ON MULTIPLE GPUS!")

This script loads the Qwen2.5-72B model, applies 4-bit quantization for efficient memory usage, and distributes the model across all available GPUs.

DeepSeek R1 (70B) and Qwen2.5-72B have extremely large checkpoints. Efficient storage and data transfer can streamline deployments:
1.	Object Storage: Store your model checkpoints in OCI Object Storage for easier retrieval from any bare metal instance.
2.	Parallel Downloads: Use tools like aria2c or multi-threaded scripts to download weights in parallel, speeding up setup time.
3.	Cache and Snapshot: After the initial model pull, create an OCI block volume or custom image snapshot that includes the downloaded model weights to avoid repeated downloads.


Fine-Tuning or Customizing Models

If you plan to fine-tune Qwen2.5 or DeepSeek R1 on your own data:
1.	Use Low-Rank Adaptation (LoRA): This can reduce the memory footprint when fine-tuning large models, saving GPU memory and training time.
2.	Experiment with Learning Rate Schedules: Large models can be sensitive to hyperparameter choices, so a careful tuning strategy can improve model accuracy without excessive training costs.
3.	Leverage OCI Data Science: OCI Data Science service can help you orchestrate distributed training using pre-built Conda environments and integrated notebook sessions.

Optimizing Performance

To maximize performance on the A100 and H100 GPUs, consider the following optimizations:
1.	Use mixed precision training and inference with torch.cuda.amp.
2.	Leverage tensor parallelism and pipeline parallelism for large models.
3.	Utilize NVIDIA’s NCCL library for efficient multi-GPU communication.
4.	Libraries like xFormers or FlashAttention can provide more efficient attention mechanisms, improving throughput for large language models.
5.	You can use activation Checkpointing - For extremely large models, you can save GPU memory by recomputing certain layers during the backward pass.
6.	Consider Gradient Accumulation - if you’re doing training or fine-tuning with limited GPU memory, accumulate gradients over multiple batches.



Benchmarking

On the H100 GPUs, you can expect significantly better performance compared to A100s. For example:
o	Qwen2.5-72B-Instruct can achieve about 8.73 tokens/s (BF16) using 2 GPUs.
o	Smaller models like Qwen2.5-7B-Instruct can process 40.38 tokens/s (BF16) on a single GPU.

H100s are expected to offer even better performance due to their enhanced architecture and higher memory bandwidth.

<<<Benchmark still going on – so I will update it with a table>>>

Monitoring and Logging

When running multi-GPU workloads, monitoring system resources is crucial:
o	GPU Utilization: Use nvidia-smi or advanced monitoring tools like Prometheus + Grafana to track GPU usage, temperature, and memory.
o	OCI Monitoring: Oracle Cloud Infrastructure provides built-in monitoring for CPU, network, and memory usage. Create custom metrics for your GPU workloads to track real-time performance.
o	Logging Tools: Use Python logging, MLflow (ready-to-deploy on OCI code), or TensorBoard to keep track of logs, training metrics, and inference speeds.

Cost Management

Running large models on multiple GPUs can become expensive. Keep these best practices in mind:
o	Use Preemptible (Spot) Instances: OCI offers preemptible capacity with lower prices. Be prepared for instance interruption, but it can significantly cut costs for non-critical or experimental workloads.
o	Scale Down When Idle: Shut down or scale down GPU nodes when not in use. OCI’s flexible billing ensures you only pay for active resources.
o	Schedule Your Jobs: Automate start/stop times for training or inference tasks, ensuring you don’t run high-performance instances when not needed.

Security and Data Governance

When deploying proprietary data or models:
o	OCI Vault: Use OCI Vault for securely managing keys, secrets, and credentials.
o	Encryption: Consider encrypting data at rest using OCI’s default encryption or your own keys.
o	Compartments and IAM: Properly set up OCI compartments and Identity and Access Management (IAM) policies to restrict user access to sensitive resources and data.



Conclusion

Running DeepSeek R1 and Qwen2.5 full models on OCI’s A100 and H100 bare metal nodes provides exceptional performance for large-scale AI workloads. By leveraging the power of 8 GPUs per node and utilizing efficient deployment strategies, you can achieve impressive inference speeds and handle complex AI tasks with ease.

Remember to monitor your resource usage and costs when running these powerful models on cloud infrastructure. OCI’s flexible pricing and high-performance GPU offerings make it an excellent choice for deploying and scaling your AI applications.

Explore Oracle Cloud’s High Performance Computing offerings for cluster networking, RDMA, and job schedulers. Investigate serving solutions like NVIDIA Triton Inference Server or Launchpad for production deployments. Check the OCI Marketplace for ready-to-deploy AI frameworks or third-party tools to speed up development.

