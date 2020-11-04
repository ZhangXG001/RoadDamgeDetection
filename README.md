# RoadDamgeDetection

  Fast and accurate road damage detection is essential for the automatization of road inspection. This paper describes our solution submitted to the Global Road Damage Detection Challenge of the 2020 IEEE International Conference on Big Data, for typical road damage detection on digital images based on deep learning. The recently proposed YOLOv4 is chosen as the baseline network, while the effects of data augmentation, transfer learning, Optimized Anchors, and their combination are evaluated. We proposed a novel road damage data generation method based on a generative adversarial network, which can generate multi-class samples with a single model. The evaluation results demonstrate the effectiveness of different tricks and their combinations on the road damage detection task, which provides a reference for practical application.
  
## Datesets

You can find the IEEE BigData 2020 Global Road Damage Detection Challenge 2020 datasets here https://github.com/sekilab/RoadDamageDetector.

## Usage

Please install darknet first. you can install darknet following the introduction of https://github.com/AlexeyAB/darknet, or you can also use docker just pull darknet image simply by ```docker pull daisukekobayashi/darknet:gpu-cv-cc75```.(recommended)

### Download the code.

```
git clone https://github.com/ZhangXG001/RoadDamgeDetection.git
```
### change path 

change the obj.data and obj.names according to the path of your dataset.

### Train

run(in docker environment):

```darknet detector train data/obj.data cfg/yolov4-custom-GRDDC.cfg yolov4.conv.137```

You can find more details of usage [here] (https://github.com/AlexeyAB/darknet)

### Test

you can test the model by:
```python
python darknet.py
```
 a .txt file will be created and the test result（the categories and coordinates of the bounding box for all test images） will be written in it.

## Citation


## Acknowledgements

  This work is partially supported by National Key R&D Program of China (2019YFB1310403), Shenzhen Science and Technology Innovation Council (JCYJ20170410171923840), and National Natural Science Foundation of China (U1613227, U1813216, 61806190).
