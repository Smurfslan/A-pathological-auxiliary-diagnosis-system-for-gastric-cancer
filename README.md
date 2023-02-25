### A pathological auxiliary diagnosis system for gastric cancer

The codes for the work of "Using Less Annotation Workload To Establish A Pathological Auxiliary Diagnosis System For Gastric Cancer" (Cell Reports Medicine)

![](./Figure/Fig1.png)
(The proposed framework for the pathological auxiliary diagnosis system for gastric cancer)



### 1. Dependencies

Please install essential dependencies.

```
libtiff==0.4.2
numpy==1.18.5
opencv-python==4.4.0.42
openslide-python==1.1.2
pandas==1.1.3
Pillow==6.2.1
scikit-image==0.17.2
scikit-learn==1.0
scipy==1.7.2
seaborn==0.12.1
segmentation-models-pytorch==0.2.0
tensorboard==2.10.1
timm==0.4.12
torch==1.9.0+cu111
torchaudio==0.9.0
torchstat==0.0.7
torchsummary==1.5.1
torchvision==0.10.0+cu111
tqdm==4.48.2
visdom==0.1.8.9
```


### 2. Usage

**(1)segmentation**

- run `./1.segmentation/main6.py` to train a segmentation for gastric cancer, and results will save in `./1.segmentation/paper/patch`

- run `./1.segmentation/test.py` to test.

**(2)convert segmented result into classification matrx**

- run `./2.convert segmented result into classification matrx/run1.sh` to Convert the slide with suffix .tif into a matrix for classification, and results will save in `./2.convert segmented result into classification matrx/matrix/big` or `./2.convert segmented result into classification matrx/matrix/small`

**(3)classification**

- run `./3.classification/K_Fold_gastric_slide_big.py` to train a classification model for surgical specimens, and results will save in `./3.classification/K_fold_model_big_new` and `./3.classification/results_big_new`

- run `./3.classification/K_Fold_gastric_slide_small.py` to train a classification model for biopsy specimens, and results will save in `./3.classification/K_fold_model_small` and `./3.classification/results_small`

- run `./3.classification/test_gastric.py` to test.

- run `./3.classification/ROC.py` to draw ROC curve.
