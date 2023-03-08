# Training
**1. Train camera-based baseline with 8 GPUs.**
```
bash run.sh ./projects/baselines/CAM-R50_img1600_128x128x10.py 8
```

**2. Train LiDAR-based baseline with 8 GPUs.**
```
bash run.sh ./projects/baselines/LiDAR_128x128x10.py 8
```

**3. Train multimodal baseline with 8 GPUs.**
```
bash run.sh ./projects/baselines/Multimodal-R50_img1600_128x128x10.py 8
```

**4. Train camera-based CONet with 8 GPUs.**
```
bash run.sh ./projects/Cascade-Occupancy-Network/CAM-R50_img1600_cascade_x4.py 8
```

**5. Train LiDAR-based CONet with 8 GPUs.**
```
bash run.sh ./projects/Cascade-Occupancy-Network/LiDAR_cascade_x4.py 8
```

**6. Train multimodal CONet with 8 GPUs.**
```
bash run.sh ./projects/Cascade-Occupancy-Network/Multimodal-R50_img1600_cascade_x4.py 8
```

# Evaluation
**Evaluation example.**
```
bash run_eval.sh $PATH_TO_CFG $PATH_TO_CKPT $GPU_NUM
```

# Visualization
**Temporarily only support saving occupancy predictions (refer to [MonoScene](https://github.com/astra-vision/MonoScene#visualization) for visualization tools)**
```
bash run_eval.sh $PATH_TO_CFG $PATH_TO_CKPT $GPU_NUM --show --show-dir $PATH
```
