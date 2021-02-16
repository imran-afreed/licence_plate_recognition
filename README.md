# Vehicle entry and exit log

The objective is to automatically recognize license number of vehicles entering and exiting a given place and keep track of them
In this approach we first recognise license plate and use this to mark timestamps and how many times it has entered and exited a place

## 1. License plate recognition
  
  To recognize license plates on vehilces, a camera will be used to take images like the one shown below
  ` 
  <p align="center">
    <img src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/one.jpg" alt="vehicle pic" width="500">
  </p>
  
  This image is fed to a Convolutional Neural Network(YoLo V4) which detects license plate in given image and outputs coordinates of bounding box in the image. These coordinates are used to crop input image to get license plate. Next step is to perform Optical Charecter Recognition on this license plate.
  
  But these images when fed to an OCR Engine(![PyTesseract](https://pypi.org/project/pytesseract/))

  
  
