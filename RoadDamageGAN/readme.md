##GAN for Data Enhancement
----------
This code is used to generate the defect of D00, D10, D20 and D40.

Useage:

###1 Dataset Preparation

    ├── main.py
    ├── cropBoundingBox.py
    ├── model
        ├── checkpoint
    ├── IEEEbigdata2020
        ├── train

###2 Corp and resize bounding box to 128*128

    python cropBoundingBox.py

The images of bounding boxes are here:

    ├── IEEEbigdata2020
        ├── data
        	├── D00
        	├── D10
        	├── D20
        	├── D40

###3 Train GAN

    python main.py

###4 Generate new images

    python main.py --phase test

New images are here:

    ├── IEEEbigdata2020
        ├── train
        	├── Czech
        		├── images_x
        	├── India
        		├── images_x
        	├── Japan
        		├── images_x

