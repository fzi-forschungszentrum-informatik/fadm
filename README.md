<a target="_blank" href="https://arxiv.org/abs/2410.08551">
  <img src="https://img.shields.io/badge/arXiv-2410.08551-red.svg" alt="arXiv Badge"/>
</a>

# FADM - Full Body Anonymization Using Diffusion Models

## Introduction
This is the official implementation for the paper "Context-Aware Full Body Anonymization using Text-to-Image Diffusion Models" presented at the ACVR 2024 workshop in conjunction with ECCV 2024.


## Getting started

- Clone this repository
- create a new conda environment
- ```pip install -r requirements.txt```
- run setup.py

## Example Usage
- All arguments are availbale using ``python anonymize.py --help``
- Single file ``python anonymize.py --file example.jpg``
- Folder ``python anonymize.py --input path/to/folder/ --output path/to/folder/``
- Add ``--out_sbs`` to output a side by side image **(original left, anonymized right)**
- Choose the model version by using ``--version v`` : $v \in \{\text{21, 20ip, rv6ip}\}$ standing for [SD 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1), [SD 2.0](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) inpainting and [Realistic Vision 6 Inpainting](https://civitai.com/models/4201?modelVersionId=245627)
- ``--ds 1.0`` shows interesting results
- ``--no_pose`` seems to work
- Use ``--highvram`` to enable model caching on the GPU when enough VRAM is available

During the first run, the program will download the needed models and dependencies!

## Multi GPU support
Only one GPU can be used at the moment, but running multiple instances to anonymize different folders in parallel can be done by adding ``--cuda_device N`` to run on GPU ``N``
