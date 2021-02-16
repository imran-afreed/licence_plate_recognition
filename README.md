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
  
  
  <p align="center">
    <img src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/cropped.png" alt="cropped image" width="250">
  </p>
  
  But these images when fed to an OCR Engine(![PyTesseract](https://pypi.org/project/pytesseract/)) are not giving any text in return, scaling image, denoising etc didn't help. So converetd image into gray scale and performed OCR there was output but the edges were present in the image so they were aslo being recognized as some charecter so the output was not totally what is required, tried scaling, converting license plate into binary image, and this reduced errors but they were still present.
  
  
<p align="center">
  <img width="250" src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/no_scaling_gray%0C.png">
  <img width="250" src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/no%20scaling%20adaptiveIHR%2096%20DA%202330:%0A%0C.png">
</p>
  
  So inorder to remove that unwanted border, countur detection was performed and the contour covering the most area was considered and a rectangular bounding box corresponding to this contour gives us exact part of license plate we want. Thus by cropping the image we will get a image as shown below.
  
<p align="center">
  <img src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/cropped.png" alt="cropped image" width="250">
</p>
  
<p align="center">
  <img src="https://github.com/imran-afreed/licence_plate_recognition/blob/master/no%20scale%20font%20increased%20border%20-r%0C.png" alt="preprocessed" width="300">
</p>

  Even this image was not giving output with PyTesseract, on scaling this image to x = 0.3 and y = 0.2 the desired output came. The scaling depends on image have but since in our application the camera and position of vehicle are almost the same every time, this shouldn't be a problem.
  
