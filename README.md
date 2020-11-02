# RoadDamgeDetection
![IMAGE](https://github.com/ZhangXG001/Real-Time-Crop-Recognition/blob/master/IMG/network.jpg)

Fast and accurate road damage detection is essential for the automatization of road inspection. This paper describes our solution submitted to the Global Road Damage Detection Challenge of the 2020 IEEE International Conference on Big Data, for typical road damage detection on digital images based on deep learning. The recently proposed YOLOv4 is chosen as the baseline network, while the effects of data augmentation, transfer learning, Optimized Anchors, and their combination are evaluated. We proposed a novel road damage data generation method based on a generative adversarial network, which can generate multi-class samples with a single model. The evaluation results demonstrate the effectiveness of different tricks and their combinations on the road damage detection task, which provides a reference for practical application.
  
## Datesets

You can find the IEEE BigData 2020 Global Road Damage Detection Challenge 2020 datasets here https://github.com/sekilab/RoadDamageDetector.

## Usage

Please install darknet first.

### Download our code.

```python
git clone https://github.com/ZhangXG001/Real-Time-Crop-Recognition.git
```

### Create .csv files of dataset.

Replace the line 13 ```path = './dataset/'+ dirname +'/'``` of csv_generator.py with ```path = './IMAGE400x300/'+ dirname +'/'```,if you use the dataset IMAGE400x300.

Replace the line 13 ```path = './dataset/'+ dirname +'/'``` of csv_generator.py with ```path = './IMAGE512x384/'+ dirname +'/'```,if you use the dataset IMAGE512x384.

You can run ``` .../python csv_generator.py ```to create .csv files of dataset.


### Train

If you want to train the model,you can run

```python
cd .../Real-Time-Crop-Recognition-master
python train-aaf.py
```
You can get the well-trained model under the folder"model1".

### Test

If you want to test the model,please modify the default input_dir of test.py(line 15),then you can run

```python
python test.py
```
You can get the test result under the folder"result1".

About the CRF code we used, you can find it [here](https://github.com/Andrew-Qibin/dss_crf). Notice that please provide a link to the original code as a footnote or a citation if you plan to use it.

## Citation




## Acknowledgements

We would like to especially thank GaoBin（[github](https://github.com/gbyy422990/salience_object_detection)）,QiBin（[github](https://github.com/Andrew-Qibin/DSS)) and Ke([github](https://github.com/twke18/Adaptive_Affinity_Fields)) for the use of part of their code.

You can also find our code on our [ihub](https://code.ihub.org.cn/projects/640).
