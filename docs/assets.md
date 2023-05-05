**Note that the mIoU is silighter higher than those in our paper (see logs for more details). Bellow are several changes in the training phase:**
- We use SyncBN instead of BN/GN, which enhances the mIoU by ~0.3.
- The total training epoch is 15 instead of 24, which significantly reduces the training time.
- The models are trained with [annotation-v0.1](./prepare_data.md) (with less occupancy artifacts).

| Subset | Checkpoint | Logs | Note |
| :---: | :---: | :---: | :---: |
| Camera-based baseline | [link](https://pan.baidu.com/s/1r0n0RpRVbZI07S9EGONZxg) (code:tlif) | [link](https://pan.baidu.com/s/1IqaML1HONdlr-l_QHl272w) (code:ahqs) | train on 8 RTX3090|
| LiDAR-based baseline | [link](https://pan.baidu.com/s/1lid7aYkeCkOCQgjDsmSuWg) (code:qdsl)| [link](https://pan.baidu.com/s/1Qrw5Ird21AgrIfm5AoQ0Yw) (code:p3ra)| train on 8 RTX3090|
| Multimodal baseline | [link](https://pan.baidu.com/s/1RJti0jlU4KxvSUBJtgoPbA) (code:d3vl)| [link](https://pan.baidu.com/s/19gsXJglRgXw8crnrMYNUTw) (code:f5qq)| train on 8 RTX3090|
| Camera-based CONet | [link](https://pan.baidu.com/s/163T31bKzPxVvYtaidvwjSA) (code:630w) | [link](https://pan.baidu.com/s/1tro5dbbaNLgXmczknDj7IQ) (code:jb9o) | train on 8 A100|
| LiDAR-based CONet | [link](https://pan.baidu.com/s/1E3lh7_hkqHcovhohuejUYg) (code:hnaf)| [link](https://pan.baidu.com/s/1ddNEGGhCK2biXeqOW0Buqg) (code:hqto) | train on 8 RTX3090|
| Multimodal CONet | [link](https://pan.baidu.com/s/1iWoZfzs12nvwoX-Ytcizvg) (code:k9p9)| [link](https://pan.baidu.com/s/1scMVRl0qmVoIB-fFI2tGMQ) (code:t7c5)| train on 8 A100|

Google Drive [link](https://drive.google.com/file/d/1rqzMy0GDsvTXNMh4Zk2WsPBFXd0jobH0/view?usp=sharing)
