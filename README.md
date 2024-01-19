# Enhancing-Low-Light-Object-Detection-with-Log-RGB-Image-Pre-processing
-------------------------------------------------------------------------------------------------------------------------------------------------------
##Description - 
The conventional signal processing pipelines and compression techniques applied to visual data often disrupt the inherent rules governing the interaction of light and matter, consequently limiting the potential of deep neural networks in learning vision-based tasks. This study delves into the notion that leveraging linear or log RGB images, which adhere more closely to the fundamental physics of reflection, might offer a more conducive input for deep networks. The goal of this project is to research whether attempting to undo the effects of existing signal processing pipelines provides a better input for deep networks for object detection.


### RAW NOD (Night Object Detection) Data set used -- https://github.com/igor-morawski/RAW-NOD/tree/main/annotations/Nikon

## Data Processing - 
### In the data processing phase, three distinct encoding techniques were employed:
1. sRGB Encoding: The sRGB images were saved in the JPEG format. These images underwent a gamma correction factor of 2.2 during the conversion process from the original raw images. This step resulted in the generation of non-linear standard RGB image data using
rawpy library.
2. Linear RGB Conversion: The initial step was to convert the raw images into linear RGB. Here, a gamma correction of 1 was applied, signifying a linear relationship between the digital values and the actual intensity of light. Following the conversion to linear RGB, the images were saved in the 16-bit TIFF(Tagged Image File Format) format. This format was chosen due to its ability to retain higher bit depths.

## OBJECT DETECTION PIPELINE USING FASTER-RCNN
### 1. Annotation Parser and Dataloader -
I created an AnnotationParser class to read JSON files following the COCO style format. This class parses ground truth boxes in xmin, ymin, xmax, and ymax format, along with their corresponding labels. Additionally, I incorporated code to scale down the bounding box coordinates when resizing images. In another class, ObjectDetectionDatasets, I read all these annotations and implemented padding for both bounding boxes and class labels. This step, combined with image resizing, enables batching of images. In a batch containing ’n’ images with varying object counts in each, I consider the maximum number of objects across any image. I pad the remaining images with ’-1’ to match this maximum length. After preprocessing the dataset, I split it into three parts: training, testing, and validation sets. The split was randomized, with 70% of the data allocated for training and 15% for both testing and validation

### 2. The Backbone Network -
we’re employing ResNet 50 as our foundational network here. In ResNet 50, each block is made up of stacks of bottleneck layers. These layers reduce the image size by half along the spatial dimension while doubling the number of channels. A bottleneck layer within ResNet 50 consists of three convolutional layers and includes a skip connection. We’re specifically utilizing the initial four blocks of ResNet 50 as our backbone network

### 3. Decoding Anchor Points -
Each location on the feature map acts as an anchor point, essentially forming an array representing coordinates across the width and height dimensions. These anchor points serve as reference positions for generating bounding boxes.

### 4. Anchor Boxes and Offsets -
Positive anchor boxes contain an object and negative anchor boxes do not. We only need to sample a few anchor boxes for training. To sample Fig. 4. Anchor Box Positive and Negative Anchor Boxes positive anchor boxes, we select the anchor boxes that have an IoU (Intersection over union) more than 0.7 with any of the ground truth boxes. To sample negative anchor boxes, we select the anchor boxes that have an IoU less than 0.3 with any of the ground truth boxes.

### 5. Building the Two-stage Model Architecture -
1. First stage - Region Proposal Network The region proposal network is the stage 1 of the detector which takes the feature map and produces region proposals. Here we combine the backbone network, the sampling module, and the proposal module into the region proposal network. During both training and inference, the RPN produces scores and offsets for all the anchor boxes. During inference, we select the anchor boxes with scores above a given threshold and generate proposals using the predicted offsets.
2. Second staeg - Classification Model: In second stage we receive region proposals, but all proposals do not have the same size. We usually resize images in image classification tasks, but the problem is resizing is not a differentiable operation, and so backpropagation cannot happen through this operation. Hence, alternative way is to divide the proposals into roughly equal subregions and apply a max pooling operation on each of them to produce outputs of same size. This is called ROI pooling.
3. Non-max suppression: Non-max suppression is basically removing duplicate bounding boxes. In this technique, we first consider the bounding box with highest classification score. Then we compute IoU of all the other boxes with this box and remove the ones which have a high IoU score. The NMS processing step is implemented in the stage 1 regression network, it’s not implemented from scratch, I used torchvision.ops library.

### Final Stage :
We put together the region proposal network and the classification module to build the final end-to-end FasterRCNN model

## EXPERIMENTS AND RESULTS
### 1. Standard RGB Train:
mAP@0.5:0.05:0.95 = 0.52645075
### 2. Linear RGB Train:
mAP@0.5:0.05:0.95 = 0.61831266
### 3. Log RGB Train:
mAP@0.5:0.05:0.95 = 0.62789305

For more results read the report - 
