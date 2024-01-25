# SegMamba
SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation

[https://arxiv.org/abs/2401.13560](https://arxiv.org/abs/2401.13560)

![](images/method_figure.jpg)

Our advantage in speed and memory.
![](images/segmamba_ablation.jpg)

M1 is the 3D UX-UNet, and M3 is SwinUNETR. They are unable to model the global features of the entire volume. In M2, the self-attention module struggles to model excessively long sequences.

## Environment install
Clone this repository and navigate to the root directory of the project.

```bash
git clone https://github.com/ge-xing/SegMamba.git

cd SegMamba
```
### Install casual-conv1d

```bash
cd casual-conv1d

python setup.py install
```

### Install mamba

```bash
cd mamba

python setup.py install
```

### Install monai 

```bash
pip install monai
```


## Acknowledgement

[https://github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI)

[https://github.com/hustvl/Vim](https://github.com/hustvl/Vim)

[https://github.com/bowang-lab/U-Mamba](https://github.com/bowang-lab/U-Mamba)


# Other awesome Mambas

1. U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation [paper
](https://arxiv.org/abs/2401.04722) | [code](https://github.com/bowang-lab/U-Mamba)

2. Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model [paper
](https://arxiv.org/abs/2401.09417) | [code](https://github.com/hustvl/Vim)

3. VMamba: Visual State Space Model [paper](https://arxiv.org/abs/2401.10166) | [code](https://github.com/MzeroMiko/VMamba)

4. Diffusion Models Without Attention [paper](https://arxiv.org/abs/2311.18257) | [code]()