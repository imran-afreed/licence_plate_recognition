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

### 2.Detection
to test the detection of license plate individually [detect.py](https://github.com/imran-afreed/licence_plate_recognition/blob/master/detect.py) can be used. Open code in any editor and change the image path appropriately and run it, this shows and saves cropped image of license plate as cropped.png. 

<p align="center">
    <img src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/images/cropped.png" alt="cropped image" width="250">
</p>
<p align="center">
  License plate cropped
</p>


### 3.pre processing image and OCR
But this images when fed to an OCR Engine([PyTesseract](https://pypi.org/project/pytesseract/) doesn't give any text in return, so converetd image into gray scale and performed OCR. This was giving output with some errors due to edges on right and left side of image. These were also recognized as some charecter by OCR engine as shown below 
  
<p align="center">
  <img width="250" src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/images/no_scaling_gray%0C.png">
</p>
<p align="center">
  Gray scale image and OCR output = " IHR 26 DA 2330 ."
</p>

Some gray scale license plates are not giving any output so converetd grayscales to binary images(only black and white) using adaptive threshold fucntion of OpenCV, and these were giving some output but the problem of border is still present

<p align="center">
    <img width="250" src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/images/no%20scaling%20adaptiveIHR%2096%20DA%202330:%0A%0C.png">
</p>
<p align="center">
  binary image and OCR output = " IHR 26 DA 2330 :"
</p>

Contour detection can be performed to detect the largest rectangle with no border in it and upon scaling to smaller size, we get desired output
  
<p align="center">
  <img src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/images/no%20scale%20font%20increased%20border%20-r%0C.png" alt="preprocessed" width="300">
</p>
<p align="center">
    border cropped and scaled, OCR output = " HR 26 DA 2330 "
</p>

[trial_and_error.py](https://github.com/imran-afreed/licence_plate_recognition/blob/master/trial_and_error.py) can be used to test and compare parameters used during preprocessing.
