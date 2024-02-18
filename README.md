# Acceleration of a classification model for thoracic diseases

## Introduction

Medical imaging is an indispensable technology for modern medicine,
and deep learning has been applied to this field since its early times.
A typical example is the modeling of reading and diagnosis of medical images such as X-rays, CTs, and MRIs.
For example, if we can construct a model that can classify diseases and estimate the location of diseases in medical images, we can expect to reduce the burden on the image reading physicians, equalize diagnostic criteria, and realize early diagnosis and prediction of disease onset beyond human capabilities.

For deep learning for medical images, it was difficult to prepare a widely shared dataset for academic research purposes, such as the *ImageNet* dataset for general object recognition, due to patient consent and privacy protection, etc.
In response, the National Institutes of Health (NIH) released datasets of chest X-ray images and CT images that are large enough for deep learning, creating an environment for developing models for clinical use.

- [*NIH Clinical Center provides one of the largest publicly available chest x-ray datasets to scientific community*, 2017.](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)
- [*NIH Clinical Center releases dataset of 32,000 CT images*, 2018.](https://www.nih.gov/news-events/news-releases/nih-clinical-center-releases-dataset-32000-ct-images)

Although medical imaging datasets are being developed, large datasets and the associated large-scale models are becoming computationally expensive for training and inference.
On the other hand, when dealing with new infectious diseases, etc., it is necessary to train models from real clinical data as early as possible,  hence the acceleration of training using *accelerators* and *parallel computing* remains an important factor.

In addition, it is also necessary to make it possible to utilize highly accurate models in medical practice at a reasonable computational cost for their widespread use.
In particular, it would be useful in terms of cost if the inference models can be applied to edge computing in clinical setting.

From the above perspective, here we have verified the reduction of the computational cost of training and inference, by accelerating the training using accelerators with a **distributed data parallel**, DDP, and by **optimization** and **quantization** of the inference model.

## Dataset

In 2017, the NIH released *ChestX-ray8*, a dataset built from over 30,000 chest X-ray images[^Wang]. This dataset was later expanded to *ChestX-ray14* by increasing the number of annotated diseases. We used this ChestX-ray14 for training.

The ChestX-ray14 dataset consists of 112,120 chest X-ray images of 30,805 patients, with multiple disease labels corresponding to each image from 14 different thoracic diseases. The labels of the ChestX-ray14 dataset are based on findings extracted from *electronic health records* by *natural language processing*, which is an example of a method to add labels based on real-world clinical diagnoses.

We trained and validated the dataset by dividing it into a training set of 70\%, a validation set of 10\%, and a test set of 20\%. The included diseases and their percentages can be output in `stats.py`. The breakdown is shown in Table 1.

|                   | **train** | **val** | **test** |
|:----------------- | --------: | ------: | -------: |
|Atelectasis        |   10.2    |   10.0  |   10.8   |
|Cardiomegaly       |    2.5    |    2.1  |    2.6   |
|Effusion           |   11.8    |   11.5  |   12.3   |
|Infiltration       |   17.7    |   18.0  |   17.6   |
|Mass               |    5.1    |    5.6  |    5.1   |
|Nodule             |    5.6    |    5.5  |    6.0   |
|Pneumonia          |    1.2    |    1.2  |    1.1   |
|Pneumothorax       |    4.7    |    4.5  |    4.9   |
|Consolidation      |    4.2    |    4.0  |    4.3   |
|Edema              |    2.2    |    1.8  |    1.8   |
|Emphysema          |    2.3    |    1.9  |    2.3   |
|Fibrosis           |    1.5    |    1.5  |    1.6   |
|Pleural Thickening |    2.9    |    3.3  |    3.3   |
|Hernia             |    0.2    |    0.4  |    0.2   |

**Table 1. Percentage of thoracic disease in each split for training, validation, and testing of the ChestX-ray14 dataset.**

## Model


For thoracic disease classification, it is possible to use models based on *convolutional neural networks*, CNN and *Vision Transformer*, here we use **CheXNet** proposed by Rajpurkar et al. in 2017[^Rajpurkar].
CheXNet is a model based on DenseNet-121, a typical CNN, and inferences multi-label classification for thoracic diseases using chest X-ray images as input. The difference from DenseNet-121 is the addition of an output layer for classification of 14 diseases.

This implementation uses an improved version of the original CheXNet model with a sigmoid function added to the final layer.

## Methods

For training CheXNet with ChestX-ray14, we load the weights of DenseNet-121 pretrained on the ImageNet dataset, and then fine-tune on the ChestX-ray14 dataset. **GPU** and **Habana Gaudi** are adopted as accelerators for training, and DDP training is performed using multiple accelerators.

First, we run a script to download the dataset.

```bash
$ bash batch_download.sh
```

## Training

In this implementation, you can choose between GPU and Habana Gaudi as the accelerators.
The Habana Gaudi hardware was tested on `dl1.24xlarge` instances provided by *Amazon Web Services*, AWS.

The following script installs the Python modules used for training and inference.

```bash
$ pip install -qr requirements.txt
```

The following commands perform DDP training using 8 Habana Gaudi accelerators.

```bash
$ torchrun --nnodes=1 --nproc_per_node=8 main.py --hpu
```

The results of training with Habana Gaudi are shown below.

```
Loading Habana modules from /usr/local/lib/python3.7/dist-packages/habana_frameworks/torch/lib
Using hpu device.
training 3750 batches 14999 images
validation 3750 batches 14999 images
epoch   1 batch  3750/ 3750 train loss 1.5167 899.075sec
epoch   2 batch  3750/ 3750 train loss 1.4969 859.069sec
...
```

As a comparison, the training time with Tesla V100 running on an AWS `p3.8xlarge` instance is shown.

```
Using cuda device.
training 3750 batches 14999 images
epoch   1 batch  3750/ 3750 train loss 1.5171 1044.471sec
epoch   2 batch  3750/ 3750 train loss 1.4974 1047.659sec
...
```

## Inference

One effective approach to reduce the computational cost of inference is to optimize and quantize the model.
If the computational cost can be reduced, edge computing using medical devices becomes feasible, and this will lower the barrier to adoption in clinical settings.
There are several implementations for inference model optimization and quantization, here we employ Intel **OpenVINO** to optimize and quantize the CheXNet model.

Since model optimization and quantization with OpenVINO is a rather complicated procedure, the procedure is summarized in `infer.sh`.
By executing this script, you can convert the PyTorch model to the optimized and quantized models.

```bash
$ bash infer.sh
```

Inference on the respective models for PyTorch, FP32 optimization, and INT8 quantization is performed by the following commands.

```bash
$ python infer.py --mode torch
$ python infer.py --mode fp32
$ python infer.py --mode int8
```

For the validation, we performed multi-label inference on the test split using the trained CheXNet model, and compared the time.
Inference is performed by performing 10 crops for each sample, and the average of the results was used to determine the predicted label.

The inference model was validated by using Intel *DevCloud for the Edge*.
As a result, we obtained a 6.2x performance improvement in the optimization of the FP32 model using OpenVINO, and a 2.6x performance improvement in the quantization to the INT8 model.
Although performance-oriented quantization was used in quantization, the ROC-AUC evaluation showed an average AUC=0.843 for FP32 to an average AUC=0.842 for INT8, indicating that even with quantization, there was only a slight loss of accuracy and no practical problem.

## Acknowledgements

Use of the ChestX-ray14 dataset was advised by Noriaki Sato, Kyoto University.

The initial implementation for OpenVINO was done by Hiroshi Ouchiyama, Intel Japan.

## References

[^Wang]: X. Wang et al., *ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*, **IEEE Conference on CVPR**, 2017.
[^Rajpurkar]: P. Rajpurkar et al, *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*, **arXiv**, 2017.
