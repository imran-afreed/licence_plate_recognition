# Vehicle entry and exit log

The objective is to automatically recognize license number of vehicles entering and exiting a given place and keep track of them
In this approach we first recognise license plate and use this to mark timestamps and how many times it has entered and exited a place

## 1. License number recognition
  
  To recognize license number of vehilces, image of car with license plate in it will be used like the one shown below
  
  <p align="center">
    <img src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/images/one.jpg" alt="vehicle pic" width="500">
  </p>
  ` 
  Using a Object detection Neural Network(YoLov4) license plate is identified and cropped out. Now the cropped image will be processed and sent to OCR engine([PyTesseract](https://pypi.org/project/pytesseract/)) which converts text in images to string.
  
### 1. Training Neural Network 
        
I have used darknet to train YoLov4, [darknet](https://github.com/pjreddie/darknet) repository is cloned and dataset is loaded into data/obj folder in the darknet folder and is uploaded [here](https://drive.google.com/file/d/1MJ3SAUATeJPNPx-eIp09OkDY_Go9G568/view?usp=sharing). 
 
 To get started with training create a folder in google drive with name license_number_recognition and upload [darknet.zip](https://drive.google.com/file/d/1MJ3SAUATeJPNPx-eIp09OkDY_Go9G568/view?usp=sharing) in it. Now open [Train.ipynb](https://github.com/imran-afreed/licence_plate_recognition/blob/master/Train.ipynb) in [google colab]( colab.research.google.com) and exectue every cell. purpose of each cell is commented in the file itself. Once training is done the weights file is saved in backup directory inside the darknet directory on google drive.
        
<p align="center">
  <img src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/images/chart_yolov4.png" alt="Training chart" width="500">
</p>
<p align="center">
  Training loss chart
</p>

### Detection
    to test the detection of license plate individually [detect.py](https://github.com/imran-afreed/licence_plate_recognition/blob/master/detect.py) can be used.













  
  <p align="center">
    <img src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/images/cropped.png" alt="cropped image" width="250">
  </p>
  
  But these images when fed to an OCR Engine([PyTesseract](https://pypi.org/project/pytesseract/)) are not giving any text in return, scaling image, denoising etc didn't help. So converetd image into gray scale and performed OCR there was output but the edges were present in the image so they were aslo being recognized as some charecter so the output was not totally what is required, tried scaling, converting license plate into binary image, and this reduced errors but they were still present.
  
  
<p align="center">
  <img width="250" src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/images/no_scaling_gray%0C.png">
  <img width="250" src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/images/no%20scaling%20adaptiveIHR%2096%20DA%202330:%0A%0C.png">
</p>
  
  So inorder to remove that unwanted border, countur detection was performed and the contour covering the most area was considered and a rectangular bounding box corresponding to this contour gives us exact part of plate we want. Thus by cropping the image we will get a image as shown below.
  
<p align="center">
  <img src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/images/no%20scale%20font%20increased%20border%20-r%0C.png" alt="preprocessed" width="300">
</p>

  Even this image was not giving output with PyTesseract, on scaling this image to x = 0.3 and y = 0.2 the desired output came. The scaling depends on image have but since in our application the camera and position of vehicle are almost the same every time, this shouldn't be a problem.
  
  ![trial and errors performed](https://github.com/imran-afreed/licence_plate_recognition/blob/master/images/flow_chart.png)
  
Code for this is in [license_plate_recognition.py](https://github.com/imran-afreed/licence_plate_recognition/blob/master/licence_plate_recognition.py) and is well commented.

### Training 
  start by downloding these files/folders
  
  [darknet.zip](https://drive.google.com/file/d/1MJ3SAUATeJPNPx-eIp09OkDY_Go9G568/view?usp=sharing)
  
  [weights](https://drive.google.com/file/d/1ZwR7HqgEVr5Rx1iyusCpthaz9W4xZQrb/view?usp=sharing)

  Create a directorty named license in google drive and upload [darknet.zip](https://drive.google.com/file/d/1MJ3SAUATeJPNPx-eIp09OkDY_Go9G568/view?usp=sharing) in it. Now open [Train.ipynb](https://github.com/imran-afreed/licence_plate_recognition/blob/master/Train.ipynb) in [google colab](https://colab.research.google.com/drive/1p0Nf1tp5bg-2-AnltzYvM9TQ55lHqaFw?authuser=1#scrollTo=5dXbN2AWv3BX) and execute all cells
  
  #### Training chart:
<p align="center">
  <img src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/images/chart_yolov4.png" alt="preprocessed" width="500">
</p>
