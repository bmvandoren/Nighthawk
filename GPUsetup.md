# GPU Setup for Nighthawk on a JupyterHub Cluster

The following instructions document how to configure a GPU-enabled Python environment for running Nighthawk on a JupyterHub cluster with an NVIDIA GPU. They were tested on a system with an NVIDIA A100-SXM4-80GB GPU, driver version 570.211.01, and CUDA 12.8.

These instructions assume you are using conda to manage your Python environments and that you have confirmed your GPU is visible to the system by running:

```
nvidia-smi
```

If `nvidia-smi` does not show your GPU, the issue is at the driver level and is outside the scope of these instructions.

## If the Standard Installation Does Not Work

On JupyterHub clusters, TensorFlow may cannot detect the GPU even when `nvidia-smi` shows it correctly. This may be because the CUDA runtime libraries that TensorFlow needs are not automatically added to `LD_LIBRARY_PATH`. TensorFlow silently falls back to CPU and print:

```
Could not find cuda drivers on your machine, GPU will not be used.
```

Additionally, installing `tensorflow[and-cuda]` (the modern bundled approach) may fail on Python 3.10 due to a TensorRT version conflict:

```
ERROR: No matching distribution found for tensorrt-libs==8.6.1
```

The instructions below work around both of these issues.

## Installation

First, create a new Python environment named `nighthawk-gpu` that uses Python 3.10:

```
conda create -n nighthawk-gpu python=3.10
```

Then activate the environment:

```
conda activate nighthawk-gpu
```

Install TensorFlow 2.15 and the required CUDA runtime libraries separately. Do not use the `[and-cuda]` extra, as it will fail due to a TensorRT dependency conflict on Python 3.10:

```
pip install tensorflow==2.15.0

pip install \
  nvidia-cudnn-cu11==8.6.0.163 \
  nvidia-cublas-cu11 \
  nvidia-cuda-runtime-cu11 \
  nvidia-cufft-cu11 \
  nvidia-curand-cu11 \
  nvidia-cusolver-cu11 \
  nvidia-cusparse-cu11
```

Then install Nighthawk and its remaining dependencies:

```
pip install nighthawk
```

---

## Configuring LD_LIBRARY_PATH

The CUDA libraries installed above are placed inside the conda environment's `site-packages` directory. TensorFlow may not be able to find them unless `LD_LIBRARY_PATH` is set to include those locations. To make this configuration automatic every time you activate the environment, run:

```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/cuda.sh << 'EOF'
SP=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $SP -name "lib" -type d | tr '\n' ':'):/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
EOF
```

After running this command, deactivate and reactivate the environment to apply the configuration:

```
conda deactivate
conda activate nighthawk-gpu
```

## Verifying the Installation

To confirm that TensorFlow can now see the GPU, run:

```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

should see output like:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

then run Nighthawk as usual:

```
nighthawk my_file.wav
```

## Warnings 

The following warnings may appear during a GPU run. They are harmless and do not affect Nighthawk's output.

**`TF-TRT Warning: Could not find TensorRT`**
TensorRT is not installed. TensorRT-based optimizations will not be used, but Nighthawk does not require them.

**`successful NUMA node read from SysFS had negative value (-1)`**
This occurs in virtualized environments that do not expose real NUMA topology. It can be safely ignored.

**`Couldn't get ptxas version` / `Failed to launch ptxas`**
The CUDA compiler toolkit is not installed system-wide on this cluster. TensorFlow falls back to the GPU driver to handle PTX compilation. There is no meaningful impact on Nighthawk performance.

To suppress the ptxas warnings, add the following line to your activation script:

```
echo 'export PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH' \
  >> $CONDA_PREFIX/etc/conda/activate.d/cuda.sh
```