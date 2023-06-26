# ArtFusion: Controllable Arbitrary Style Transfer using Dual Conditional Latent Diffusion Models<br><sub>Official PyTorch Implementation</sub>

[arXiv](https://arxiv.org/abs/2306.09330) | [BibTeX](#bibtex)

*Author: Dar-Yen Chen*

This implementation is based on the [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) repository.

<p align="center">
<img src=assets/results.gif />
</p>

Our paper presents the first learning-based arbitrary style transfer diffusion model. 
ArtFusion exhibits outstanding controllability and faithful representation of artistic details.

## Controllability

<p align="center">
<img src=assets/one_dim.png />
<img src=assets/two_dim.png width="80%"/>
</p>

ArtFusion empowers users with the flexibility to balance between source content and reference style in the outputs, catering to diverse stylization preferences.
Results range from distinct content structures to pronounced stylization.

## Style Representation

<p align="center">
<img src=assets/compares_detail.png />
</p>

ArtFusion can capture the core style characteristics that are typically overlooked in SOTA methods, 
such as the blurry edges typical of Impressionist art, the texture of oil painting, and similar brush strokes.

## Architecture

<p align="center">
<img src=assets/artfusion.png />
</p>

## Environment
Create and activate the conda environment:

```
conda env create -f environment.yaml
conda activate artfusion
```

## Training

The WikiArt style dataset we use is from [Kaggle](https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan),
which is gathered from [WIKIART](https://www.wikiart.org/).

The content dataset is [MS COCO 2017](https://cocodataset.org/).

Please download and place the datasets as:
```bash
└── datasets
    ├── ms_coco
    └── wiki-art
```

Download the [first-stage VAE](https://ommer-lab.com/files/latent-diffusion/kl-f16.zip) utilized in LDM to `./checkpoints/vae/kl-f16.ckpt`.

Then run the commands:

```
python main.py \
    --name experiment_name \
    --base ./configs/kl16_content12.yaml \
    --basedir ./checkpoints \
    -t True \
    --gpus 0,
```

## Pretrained Model
The pretrained model can be downloaded [here](https://1drv.ms/u/s!AuZJlZC8oVPfgWC2O77TUlhIfELG?e=RoSa8a).

Please place it at the folder `./checkpoints/artfusion/`.

## Inference
Inference can be done via the [notebook](notebooks/style_transfer.ipynb).

Type following to set the conda environment on the jupyter notebook.

```
python -m ipykernel install --user --name artfusion
```

## License

This project is released under the [MIT License](LICENSE).

## BibTeX
If you find this repository useful for your research, please cite using the following.

```
@misc{chen2023artfusion,
      title={ArtFusion: Controllable Arbitrary Style Transfer using Dual Conditional Latent Diffusion Models}, 
      author={Dar-Yen Chen},
      year={2023},
      eprint={2306.09330},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

