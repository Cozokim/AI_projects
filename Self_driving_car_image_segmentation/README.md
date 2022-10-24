# :dart: Image segmentation for autonomous vehicles
The goal of this project is to design the onboard **image segmentation** part of a computer vision system for autonomous vehicles.

**Image segmentation** is the process of classifying each pixel in an image to a particular class (or label). It can be thought of as a classification problem per pixel, but it doesn't distinguish between different instances of the same object; for example, there could be multiple cars in the scene and all of them would have the same label.

# :card_index_dividers: Dataset
[Cityscapes Dataset Overview](https://www.cityscapes-dataset.com/dataset-overview/)

<img src='/static\cityscapes_dataset.png'>

# :scroll: Tasks
- :heavy_check_mark: Structure the dataset into relevant directories (train, val, test);
- :heavy_check_mark: Perform Exploratory Data Analysis;
- :heavy_check_mark: Prepare images processing to be able to segment masks from 30 to 8 main labels;
- :heavy_check_mark: Generate batches of images (Data Generation);
- :heavy_check_mark: Choose relevant metrics (and losses);
- :heavy_check_mark: Train and evaluate different models, from baseline to advanced (including data augmentation);
- :heavy_check_mark: Deploy model as Flask API on Microsoft Azure.

# :computer: Dependencies
- <code>pip install tensorflow opencv-python flask</code>

# :pushpin: References 
- [Official CityScapes Github - Marius Cordts](https://github.com/mcordts/cityscapesScripts) and its [labels' management](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py);
- Heavy data manipulation : [Use of 'Sequence' class with keras](https://keras.io/api/utils/python_utils/#sequence-class), ['Sequence' class with tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence), [Afshine and Shervice Amidi](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly), [Bastien Maurice](https://deeplylearning.fr/cours-pratiques-deep-learning/realiser-son-propre-generateur-de-donnees/)
- Data augmentation with [imgaug](https://imgaug.readthedocs.io/en/latest/) or [albumentations](https://albumentations.ai/)
- GitHub references for this project : [Divam Gupta](https://github.com/divamgupta/image-segmentation-keras); [Pavel Yakubowskiy](https://github.com/qubvel/segmentation_models); [Malte Koche](https://github.com/baudcode/tf-semantic-segmentation)
- arxiv.org and paperswithcode.com : [Fully Convolutional Network (FCN)](https://nanonets.com/blog/how-to-do-semantic-segmentation-using-deep-learning/); [Unet](https://arxiv.org/pdf/1505.04597v1.pdf); [SegNet](https://arxiv.org/pdf/1511.00561v3.pdf); [Cityscapes benchmark from 2015 to 2021](https://paperswithcode.com/sota/real-time-semantic-segmentation-on-cityscapes)
- Articles : [2021 guide to Semantic Segmentation - Nanonets](https://nanonets.com/blog/semantic-image-segmentation-2020/), [Beginner's guide by Divam Gupta](https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html)
- Other resources : [Simple guide - Bharath Raj](https://medium.com/beyondminds/a-simple-guide-to-semantic-segmentation-effcf83e7e54), [Popular architectures - Priya Dwivedi](https://towardsdatascience.com/semantic-segmentation-popular-architectures-dff0a75f39d0), [Keras Pipeline - Rwiddhi Chakraboty](https://towardsdatascience.com/a-keras-pipeline-for-image-segmentation-part-1-6515a421157d)
- Explanation of semantic segmentation metrics and loss functions : [Intersection over union](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/); [Loss function with Keras and PyTorch](https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch)
- Flask API deployment on Azure : [Microsoft Docs](https://docs.microsoft.com/fr-fr/azure/app-service/quickstart-python?tabs=bash&pivots=python-framework-flask)
