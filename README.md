# ArcFace-MXNet-Gluon
ArcFace with MXNet Gluon
## Requirements
+ OpenCV 4.1
+ scikit-image 1.15
+ Dlib 19.17
+ Tensorflow 1.14
+ MXNet 1.5
## Installation
### MMOD Face Detection
```bash
mkdir -p shared/MMOD                    
# Download the dlib face detection model
wget http://dlib.net/files/mmod_human_face_detector.dat.bz2 -P shared/MMOD
# Unpack it
bzip2 -d shared/MMOD/mmod_human_face_detector.dat.bz2
```
### PRNet Landmark Detection
1. Download the pre-trained model shared by the PRNet authors
 at [GoogleDrive](https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view?usp=sharing)
2. Copy it to **_shared/PRNet/net-data_**

## Data Preparation
### Training Data
**Deep Glint Dataset**
> http://trillionpairs.deepglint.com/data

+ The Deep Glint dataset is a combination of MS-Celeb-1M-v1c and Asian-Celeb datasets.
It contains around 6.75M images of 180,855 subjects.

  |# Images |# Detected Faces|# IDs|
  |--------:|---------------:|----:|
  |6,753,545|6,749,639|180,855|
+ Download two files **_train_msra.tar.gz_** (125GB) and **_train_celebrity.tar.gz_** (91GB)
and unzip both of them to a directory, e.g., **_/mnt/Datasets/Glint/images_**
+ Pre-processing
  ```bash
  # Create a list of images
  python data/dir2lst.py -i /mnt/Datasets/Glint/images -o /mnt/Datasets/Glint/glint.lst
  # Run MMOD face detection
  python MMOD/face_detector.py -p 8 -d /mnt/Datasets/Glint/images -i /mnt/Datasets/Glint/glint.lst
  # Run PRNet landmark detection
  python PRNet/landmark_detector.py -d /mnt/Datasets/Glint/images -i /mnt/Datasets/Glint/glint_dlib.lst
  # Run face2rec to create a record file for mxnet
  python data/face2rec.py /mnt/Datasets/Glint/glint_dlib_prnet /mnt/Datasets/Glint/images --pack-label --num-thread 8
  ```
## Licence
+ Our code is released under MIT License.
## Credits
Many parts of our code is copied from:
+ [PRNet](https://github.com/YadiraF/PRNet)
+ [InsightFace](https://github.com/deepinsight/insightface)