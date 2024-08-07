# UnderwaterSAM2Eval

This repository is the implementation of "[Evaluation of Segment Anything Model 2: The Role of SAM2 in the Underwater Environment](https://arxiv.org/abs/2408.02924)".

If you found this project useful, please give us a star ⭐️ or [cite](#citation) us in your paper, this is the greatest support and encouragement for us.

## Abstract

With breakthroughs in large-scale modeling, the [Segment Anything Model](https://github.com/facebookresearch/segment-anything) (SAM) and its extensions have been attempted for applications in various underwater visualization tasks in marine sciences, and have had a significant impact on the academic community. Recently, Meta has further developed the [Segment Anything Model](https://github.com/facebookresearch/segment-anything-2/tree/main) 2 (SAM2), which significantly improves running speed and segmentation accuracy compared to its predecessor. 
This report aims to explore the potential of SAM2 in marine science by evaluating it on the underwater instance segmentation benchmark datasets UIIS and USIS10K. The experiments show that the performance of SAM2 is extremely dependent on the type of user-provided prompts. When using the ground truth bounding box as prompt, SAM2 performed excellently in the underwater instance segmentation domain. However, when running in automatic mode, SAM2's ability with point prompts to sense and segment underwater instances is significantly degraded.
It is hoped that this paper will inspire researchers to further explore the SAM model family in the underwater domain.

##  Experimental Results
In this technical report, we use the underwater instance segmentation task as a case study to analyze the performance of SAM2 in underwater scenarios with the UIIS dataset (ICCV'23) and the USIS10K dataset (ICML'24). We observe the following two points:
* When ground truth is used as the prompt for the SAM2, its performance improves significantly compared to SAM and [EfficientSAM](https://github.com/yformer/EfficientSAM).\
* When using SAM2 to automatically generate instance masks, the performance of SAM2 showed significant degradation and is not comparable to state-of-the-art underwater instance segmentation algorithms.


## Conclusion

In this work, we conduct a preliminary investigation of the performance for SAM2 in the field of underwater segmentation. Based on experiments on the UIIS dataset and the USIS10K dataset, we observe:
* The performance of SAM2 is largely dependent on the type and quality of the input prompts, and when the type of prompts is constant, the difference in performance between different backbones of SAM2 is not significant. 
*  When automated inference without user-specified prompt, the performance of SAM2 shows a significant degradation. Therefore, how to design a reliable object detection module as a prompt generator for SAM2 will be the focus of future research in this area.

## Citation
If you find our repo or USIS10K dataset useful for your research, please cite us:
```
@article{
  lian2024evaluationsegmentmodel2,
  title     = {Evaluation of Segment Anything Model 2: The Role of SAM2 in the Underwater Environment}, 
  author    = {Shijie Lian and Hua Li},
  year      = {2024},
  journal   = {arXiv preprint arXiv:2408.02924},
  URL       = {https://arxiv.org/abs/2408.02924}, 
}
```
In addition, if you find the [UIIS](https://github.com/LiamLian0727/WaterMask) dataset and the [USIS10K](https://github.com/LiamLian0727/USIS10K) dataset helpful to your work, please cite them:
```
@InProceedings{
  lian2023waterMask,
  title     = {WaterMask: Instance Segmentation for Underwater Imagery},
  author    = {Lian, Shijie and Li, Hua and Cong, Runmin and Li, Suqi and Zhang, Wei and Kwong, Sam},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023},
  pages     = {1305-1315}
}
@inproceedings{
  lian2024diving,
  title     = {Diving into Underwater: Segment Anything Model Guided Underwater Salient Instance Segmentation and A Large-scale Dataset},
  author    = {Shijie Lian and Ziyi Zhang and Hua Li and Wenjie Li and Laurence Tianruo Yang and Sam Kwong and Runmin Cong},
  booktitle = {Forty-first International Conference on Machine Learning},
  year      = {2024},
  URL       = {https://openreview.net/forum?id=snhurpZt63}
}
```
