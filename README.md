<div align="center">

# Adaptive Multi-Modal Cross-Entropy Loss for Stereo Matching
</div>

<h3 align="center">
  <a href="https://arxiv.org/abs/2306.15612">arXiv</a> |
  <a href="https://xxxupeng.github.io/projects/cvpr2024/poster.pdf">Poster</a> |
  <a href="https://xxxupeng.github.io/projects/cvpr2024/demo.mp4">Video</a>
</h3>

## Abstract
Despite the great success of deep learning in stereo matching, recovering accurate disparity maps is still challenging. Currently, L1 and cross-entropy are the two most widely used losses for stereo network training. Compared with the former, the latter usually performs better thanks to its probability modeling and direct supervision to the cost volume. However, how to accurately model the stereo ground-truth for cross-entropy loss remains largely under-explored. Existing works simply assume that the ground-truth distributions are uni-modal, which ignores the fact that most of the edge pixels can be multi-modal. In this paper, a novel adaptive multi-modal cross-entropy loss (ADL) is proposed to guide the networks to learn different distribution patterns for each pixel. Moreover, we optimize the disparity estimator to further alleviate the bleeding or misalignment artifacts in inference. Extensive experimental results on public datasets show that our method is general and can help classic stereo networks regain state-of-the-art performance. In particular, GANet with our method ranks $1^{st}$ on both the KITTI 2015 and 2012 benchmarks among the published methods. Meanwhile, excellent synthetic-to-realistic generalization performance can be achieved by simply replacing the traditional loss with ours.

## Additional Experimental Results
In addition to the three baselines (PSMNet, GwcNet, and GANet) reported in the paper, we also retrained [PCWNet](https://github.com/gallenszl/PCWNet) and tested on the SceneFlow test set and KITTI 2015 benchmark. The results are as follows:

| SceneFlow | EPE | 1px | 3px |
|---|---|---|---|
| | 0.57 | 4.29 | 1.95 |

| KITTI 2015 | D1-bg | D1-fg | D1-all |
|---|---|---|---|
| |  1.39 | 2.64 | 1.60 |


## Environment
- python == 3.9.12
- pytorch == 1.11.0
- torchvision == 0.12.0
- numpy == 1.21.5
- apex == 0.1


## Datasets

- [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [KITTI 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
- [KITTI 2012](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)
- [Middlebury](https://vision.middlebury.edu/stereo/data/)
- [ETH3D](https://www.eth3d.net/datasets)

Download the datasets, and change the `datapath` args. in `./scripts/sceneflow.sh` or `./scripts/kitti.sh`.

## Pretrained Models

Pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1ygvko430bkbL25HIQFOD_0987FRj1GuD?usp=drive_link).


## Training

We use the Distributed Data Parallel (DDP) to train the model.

Please execute the bash shell in `./scripts/`, as:

```bash
/bin/bash ./scripts/sceneflow.sh
/bin/bash ./scripts/kitti.sh
```

Training logs are saved in `./log/`.

Change `loss_func` args. for different losses:
- SL1: smooth L1 loss
- ADL: **AD**aptive multi-modal cross-entropy **L**oss


If you want to train the GANet, please install the [NVIDIA-Apex package](https://github.com/NVIDIA/apex) and compile the [GANet libs](https://github.com/feihuzhang/GANet).

## Evaluation

Please uncomment and execute `val.py` in the shell scripts.

`EPE`, `1px`, `2px`, `3px`, `D1`, `4px`, `speed` are reported.

Change `estimator` args. for different disparity estimators:
- softargmax: soft-argmax
- argmax: argmax
- SME: **S**ingle-**M**odal disparity **E**stimator
- DME: **D**ominant-**M**odal disparity **E**stimator


## To Do List
Currently, the code of dataloader and evaluation are based on [PSMNet](https://github.com/JiaRenChang/PSMNet), and DDP is based on [DSGN](https://github.com/dvlab-research/DSGN). A convenient `stereo toolbox` is coming soon to support multiple dataloaders, stereo backbones, loss functions, disparity estimators, and the performance or generalization evaluation, with just a few lines of code.

## Citation
```
@InProceedings{Xu_2024_CVPR,
    author    = {Xu, Peng and Xiang, Zhiyu and Qiao, Chengyu and Fu, Jingyun and Pu, Tianyu},
    title     = {Adaptive Multi-Modal Cross-Entropy Loss for Stereo Matching},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {5135-5144}
}
```

## Acknowledgement

This project is based on the [PSMNet](https://github.com/JiaRenChang/PSMNet), [GwcNet](https://github.com/xy-guo/GwcNet), [GANet](https://github.com/feihuzhang/GANet), and [DSGN](https://github.com/dvlab-research/DSGN), we thank the original authors for their excellent works.
