# Step-by-step installation instructions


**1. Create a conda virtual environment and activate it.**
```shell
conda create -n OpenOccupancy python=3.7 -y
conda activate OpenOccupancy
```

**2. Install PyTorch and torchvision (tested on torch==1.10.1 & cuda=11.3).**
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

**3. Install gcc>=5 in conda env.**
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**4. Install MMCV following the [official instructions](https://github.com/open-mmlab/mmcv).**
```shell
pip install mmcv-full==1.4.0
```

**5. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**6. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```

**7. Install other dependencies.**
```shell
pip install timm
pip install open3d-python
pip install PyMCubes
pip install spconv-cu113
pip install fvcore
pip install setuptools==59.5.0
```

**8. Install occupancy pooling.**
```shell
git clone https://github.com/JeffWang987/OpenOccupancy.git
cd OpenOccupancy
export PYTHONPATH=“.”
python setup.py develop
```
