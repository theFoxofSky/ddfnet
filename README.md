# Decoupled Dynamic Filter Networks

This repo is the official implementation of "Decoupled Dynamic Filter Networks".

## Introduction
DDF is an alternative of convolution which decouples dynamic filters into spatial and channel filters.

![DDF operation](http://thefoxofsky.github.io/images/ddf_1.png)

We illustrate the DDF operation and the DDF module. The orange color denotes spatial dynamic filters 
/ branch, and the green color denotes channel dynamic filters / branch. The filter application means applying 
the convolution operation at a single position. ‘GAP’ means the global average pooling and ‘FC’ denotes the fully connected layer.

Please refer to our [project page](https://thefoxofsky.github.io/project_pages/ddf) and [paper](https://arxiv.org/abs/2104.14107) for more details.

## Model zoo

Will be avaliable soon.

## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/theFoxofSky/ddfnet.git
cd ddfnet
```

- Create a conda virtual environment and activate it:

```bash
conda create -n ddfnet python=3.7 -y
conda activate ddfnet
```

- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```

- Install `timm==0.4.5`:

```bash
pip install timm==0.4.5
```

- Install `Apex`:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

- Install other requirements:

```bash
pip install pyyaml ipdb
```
- Build the ddf operation:

```bash
cd ddf
python setup.py install
mv build/lib*/* .
```

- Verify the ddf operation:

```bash
cd <path_to_ddfnet>
python grad_check.py
```

### Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. Please prepare it under the following file structure:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```

### Training from scratch

To train a model, for example `ddf_mul_resnet50`, on ImageNet from scratch with 8 RTX 2080Ti, run:

```bash
./distributed_train.sh 8 <path_to_imagenet> --model ddf_mul_resnet50 --lr 0.4 \
--warmup-epochs 5 --epochs 120 --sched cosine -b 128 -j 6 --amp --dist-bn reduce
```

### Evaluation

To evaluate a pre-trained model, for example `ddf_mul_resnet50`, on ImageNet val, run:

```bash
python validate.py <path_to_imagenet> --model ddf_mul_resnet50 --checkpoint <path_to_checkpoint>
```

### Inference time

To measure the inference time, run:

```bash
python test_time.py
```

## Use ddf in other places as a basic building layer

Please directly copy the ddf folder to your repo and rebuild the ddf operation following the instructions above.
Then, you can easily import the ddf operation, the DDFPack, and the DDFUpPack. 

You can design your own module with the ddf operation.

For example, you can get a carafe/involution-like module by fixing all values in the channel filter to 1 for 'mul' combination or 0 for 'add' combination.

```python
channel_filter = torch.ones(filter_size)
output = ddf(input, channel_filter, spatial_filter,
             kernel_size, dilation, stride, head, 'mul')
```
or

```python
channel_filter = torch.zeros(filter_size)
output = ddf(input, channel_filter, spatial_filter,
             kernel_size, dilation, stride, head, 'add')
```

Similarly, you can get a WeightNet-like depthwise filter by fixing all values in the spatial filter to 1 for 'mul' combination or 0 for 'add' combination.


```python
spatial_filter = torch.ones(filter_size)
output = ddf(input, channel_filter, spatial_filter,
             kernel_size, dilation, stride, head, 'mul')
```
or

```python
spatial_filter = torch.zeros(filter_size)
output = ddf(input, channel_filter, spatial_filter,
             kernel_size, dilation, stride, head, 'add')
```

Almost all exisitng weight-dynamic depthwise operation (not grid-dynamic like deformable convolution) can be implemented with our ddf operation. Have fun exploring.

## Acknowledgement

Codebase from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models).

## Citation

If you find this code useful for your research, please cite our paper.

```
@inproceedings{zhou_ddf_cvpr_2021,
               title = {Decoupled Dynamic Filter Networks},
               author = {Zhou, Jingkai and Jampani, Varun and Pi, Zhixiong and Liu, Qiong and Yang, Ming-Hsuan},
               booktitle = {IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)},
               month = jun,
               year = {2021}
               }
```
