import cv2
import os
from tqdm import tqdm
from pathlib import Path
from shutil import copyfile

def xmlData(name, width, height, label):
    return '''<annotation>
    <folder>JPEGImages</folder>
    <filename>%s.jpg</filename>
    <path>%s.jpg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>1</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>0</xmin>
            <ymin>0</ymin>
            <xmax>1</xmax>
            <ymax>1</ymax>
        </bndbox>
    </object>
</annotation>''' % (name, name, width, height, label)

def videoCapture(video_path, save_path, label, fps=24):
    vc = cv2.VideoCapture(video_path) #读入视频文件 
    c = 1
    fc = int(vc.get(7))
    frame_count = 1

    if not vc.isOpened(): #判断是否正常打开 
        return

    JPEGImages_path = os.path.join(save_path, "JPEGImages")
    Annotations_path = os.path.join(save_path, "Annotations")
    ImageSets_path = os.path.join(save_path, "ImageSets/Main")

    if not os.path.exists(JPEGImages_path):
        os.makedirs(JPEGImages_path)
    
    if not os.path.exists(Annotations_path):
        os.makedirs(Annotations_path)
    
    if not os.path.exists(ImageSets_path):
        os.makedirs(ImageSets_path)

    with open(os.path.join(ImageSets_path, 'test.txt'), 'w') as f:
        for c in tqdm(range(fc)):
            rval, frame = vc.read()
            if not rval:
                break
            if c % fps == 0: #每隔timeF帧进行存储操作 
                cv2.imwrite(os.path.join(JPEGImages_path, "v-%06d.jpg" % frame_count), frame) #存储为图像 
                with open(os.path.join(Annotations_path, 'v-%06d.xml' % frame_count), 'w') as f_xml:
                    f_xml.write(xmlData('v-%06d' % frame_count, 0, 0, label))
                f.write("v-%06d\n" % frame_count)
                frame_count += 1
    vc.release()

def imageCapture(image_path, save_path, label):
    p = Path(image_path)
    JPEGImages_path = os.path.join(save_path, "JPEGImages")
    Annotations_path = os.path.join(save_path, "Annotations")
    ImageSets_path = os.path.join(save_path, "ImageSets/Main")

    if not os.path.exists(JPEGImages_path):
        os.makedirs(JPEGImages_path)
    
    if not os.path.exists(Annotations_path):
        os.makedirs(Annotations_path)
    
    if not os.path.exists(ImageSets_path):
        os.makedirs(ImageSets_path)
    with open(os.path.join(ImageSets_path, 'test.txt'), 'w') as f:
        for file in list(p.glob('*.jpg')):
            copyfile(file, os.path.join(JPEGImages_path, file.name))
            with open(os.path.join(Annotations_path, '%s.xml' % file.stem), 'w') as f_xml:
                f_xml.write(xmlData('%s' % file.stem, 0, 0, label))
            f.write("%s\n" % file.stem)

def videoSave(frameList, save_path, fps=24):
    size = (512, 512)
    #可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(save_path, fourcc, fps, size) #最后一个是保存图片的尺寸

    for frame in tqdm(frameList):
        videoWriter.write(frame)
    videoWriter.release()

if __name__ == '__main__':
    videoCapture('./videos/DJI_0011.MOV', './captures/DJI_0011', 'gauge')