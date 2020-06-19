### 代码目录

videos文件夹存放测试视频

captures文件夹存放提取视频帧（自动生成）

data、layers、netModel、utils文件夹存放系统文件

eval文件夹存放检测目标txt（自动生成）

useful_weight文件夹存放网络模型参数

results文件夹存放输出视频

### 运行方式

#### 违章建筑

##### 输入为视频
python visualTest_building.py --video_path=视频路径 --use_image="False"

##### 输入为图片
python visualTest_building.py --video_path=存放图片的文件夹

#### 水位尺

##### 输入为视频
python visualTest_gauge.py --video_path=视频路径 --use_image="False"

##### 输入为图片
python visualTest_gauge.py --video_path=存放图片的文件夹

`更多参数设置可以查看代码中的args`
