<div align="center">   

# OpenOccupancy: A Large Scale Benchmark for Surrounding Semantic Occupancy Perception
</div>

# Abstract 

Semantic occupancy perception is essential for autonomous driving, as automated vehicles require a fine-grained perception of the 3D urban structures. However, existing relevant benchmarks lack diversity in urban scenes, and they only evaluate front-view predictions. Towards a comprehensive benchmarking of surrounding perception algorithms, we propose OpenOccupancy, which is the first surrounding semantic occupancy perception benchmark. In the OpenOccupancy benchmark, we extend the large-scale nuScenes dataset with dense semantic occupancy annotations. Previous annotations rely on LiDAR points superimposition, where some occupancy labels are missed due to sparse LiDAR channels. To mitigate the problem, we introduce the Augmenting And Purifying (AAP) pipeline to ~2x densify the annotations, where ~4000 human hours are involved in the labeling process. Besides, camera-based, LiDAR-based and multi-modal baselines are established for the OpenOccupancy benchmark. Furthermore, considering the complexity of surrounding occupancy perception lies in the computational burden of high-resolution 3D predictions, we propose the Cascade Occupancy Network (CONet) to refine the coarse prediction, which relatively enhances the performance by ~30% than the baseline. We hope the OpenOccupancy benchmark will boost the development of surrounding occupancy perception algorithms.

[arXiv](https://arxiv.org/abs/2303.03991) 



# News
- **[2023/03/22]** The pretrained models and logs are available (see [Assets](docs/assets.md)).
- **[2023/03/20]** We have released annotation v0.1 (more lightweght and less artifacts). Check [Prepare Dataset](docs/prepare_data.md) for more details.
- **[2023/03/19]** We have updated model configs (using syncbn enhances the performance).


# Getting Started

- [Installation](docs/install.md) 

- [Prepare Dataset](docs/prepare_data.md)

- [Training, Evaluation, Visualization](docs/trainval.md)


# Demo
**Semantic Occupancy Annotations**

https://user-images.githubusercontent.com/49095445/223448290-53732e07-9642-429c-9cc4-b8ca5a967956.mp4

**Visualization of different baselines**
<div align="center">   
  
https://user-images.githubusercontent.com/49095445/223449450-0a96ed5d-b35e-4d76-8737-fea02f99f113.mp4
  
  
</div>

**Visualization of CONet**

https://user-images.githubusercontent.com/49095445/223448374-6317a8d9-4eb8-4ef8-939e-ac52982149b4.mp4


**Baseline framwork**
![occupancy-baselines](https://user-images.githubusercontent.com/49095445/223452540-111a3d96-44ed-475f-a3c1-c033b36941c6.png)

**Cascade-Occupancy-Network framwork**
![occupancy-conet](https://user-images.githubusercontent.com/49095445/223452679-cb8fdd95-5102-4563-adda-ef3645d48353.png)

# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{wang2023openoccupancy,
  title={Openoccupancy: A large scale benchmark for surrounding semantic occupancy perception},
  author={Wang, Xiaofeng and Zhu, Zheng and Xu, Wenbo and Zhang, Yunpeng and Wei, Yi and Chi, Xu and Ye, Yun and Du, Dalong and Lu, Jiwen and Wang, Xingang},
  journal={arXiv preprint arXiv:2303.03991},
  year={2023}
}
```

# Acknowledgement

Many thanks to these excellent projects:
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
- [BEVerse](https://github.com/zhangyp15/BEVerse)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)

Related Work in Occupancy Perception:
- [TPVFormer](https://github.com/wzzheng/TPVFormer)
- [SurroundOcc](https://github.com/weiyithu/SurroundOcc)
- [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction)
- [occupancy-for-nuscenes](https://github.com/FANG-MING/occupancy-for-nuscenes)
