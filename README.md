# RoadDamgeDetection
![IMAGE](https://github.com/ZhangXG001/Real-Time-Crop-Recognition/blob/master/IMG/network.jpg)

In this work, a novel strategy for crop recognition in fields with high weed infestation levels is devised based on visual attention mechanism. It detects crop plants based on their saliency in the images, which is invariant to species of crops and weeds, and thus more generalized than features like color, shape and texture. The saliency of crop plants over surrounding weeds has been demonstrated in our experiment that the CNN model trained with a general saliency detection dataset (MSRA-B, containing no field images) can detect crop plants in general.

 A deep CNN framework is proposed to detect salient regions (crop plants) in field images based on the DSS architecture proposed by Hou et al (see reference [28] in our paper). The network possesses side-output structures with short connections for extracting multi-scale features, while taking advantage of Adaptive Affinity Fields to improve the segmentation at boundaries and fine structures. It is lightweight, efficient, and able to accurately segment crop plants from weedy backgrounds.
 
A field image dataset, CWF-788 (CWF is short for crop in weedy field), is created to train and evaluate the proposed method. It contains 788 images captured from cauliflower fields with high weed pressure. High-quality pixel-wise annotated label images are provided with the dataset. This dataset can be used for future researches and fair comparisons.

## Datesets

We provide our CWF-788 dataset with two different resolutions,IMAGE400x300 with the resolution of 400x300 and IMAGE512x384 with the resolution of 512x384.

## Usage

Please install Tensorflow and required packages first.

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

If you think this work is helpful, please cite

Nan Li, Xiaoguang Zhang, Chunlong Zhang, Huiwen Guo, Zhe Sun, and Xinyu Wu, “Real-time crop recognition in transplanted fields with prominent weed growth: a visual-attention-based approach,” IEEE Access, 2019, 7(1): 185310-185321, DOI: 10.1109/ACCESS.2019.2942158


## Acknowledgements

We would like to especially thank GaoBin（[github](https://github.com/gbyy422990/salience_object_detection)）,QiBin（[github](https://github.com/Andrew-Qibin/DSS)) and Ke([github](https://github.com/twke18/Adaptive_Affinity_Fields)) for the use of part of their code.

You can also find our code on our [ihub](https://code.ihub.org.cn/projects/640).
