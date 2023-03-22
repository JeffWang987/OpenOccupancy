**Note that the mIoU is silighter higher than those in our paper (see logs for more details). Bellow are several changes in the training phase:**
- We use SyncBN instead of BN/GN.
- The total training epoch is 15 instead of 24, which significantly reduces the training time.
- The models are trained with [annotation-v0.1](./prepare_data.md) (with less occupancy artifacts).

| Subset | Checkpoint | Logs | Note |
| :---: | :---: | :---: | :---: |
| Camera-based baseline | [link](TODO) | [link](TODO) | train on 8 RTX3090|
| LiDAR-based baseline | [link](TODO) | [link](TODO) | train on 8 RTX3090|
| Multimodal baseline | [link](TODO) | [link](TODO) | train on 8 RTX3090|
| Camera-based CONet | [link](TODO) | [link](TODO) | train on 8 A100|
| LiDAR-based CONet | [link](TODO) | [link](TODO) | train on 8 RTX3090|
| Multimodal CONet | [link](TODO) | [link](TODO) | train on 8 A100|

