# coding: UTF-8
import cv2
import os
from xml.etree import ElementTree


data_path = './IEEEbigdata2020/'
for gov in ['Czech', 'India', 'Japan']:
    file_list = os.listdir(data_path + 'train/' + gov + '/annotations/xmls')

    for file in file_list:

        im_name = file.split('.')[0] + '.jpg'

        full_impath = data_path + gov + '/images/' + im_name

        infile_xml = open(data_path + gov + '/annotations/xmls/' + file, encoding='utf-8')
        tree = ElementTree.parse(infile_xml)
        root = tree.getroot()
        
        cnt = 0
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            xmlbox = obj.find('bndbox')
            xmin = int(xmlbox.find('xmin').text)
            xmax = int(xmlbox.find('xmax').text)
            ymin = int(xmlbox.find('ymin').text)
            ymax = int(xmlbox.find('ymax').text)

            if xmin>xmax:
                xmin = int(xmlbox.find('xmax').text)
                xmax = int(xmlbox.find('xmin').text)

            if ymin>ymax:
                ymin = int(xmlbox.find('ymax').text)
                ymax = int(xmlbox.find('ymin').text)

            # open image
            img = cv2.imread(full_impath)

            # crop bounding box
            roi = img[ymin:ymax, xmin:xmax]

            if roi.sum() != 0:
                # resize image
                h = 128
                w = 128
                roiResized = cv2.resize(roi, (h, w))


                im_name = im_name.split('.')[0] + '_' + str(cnt) + '.png'
               # im_name = im_name.split('.')[0] + '.png'

                # save resized image
                save_dir = data_path + 'data/' + cls_name
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(save_dir + '/' + im_name, roiResized)

                cnt = cnt + 1
                 
                im_name = file.split('.')[0] + '.jpg'
                
