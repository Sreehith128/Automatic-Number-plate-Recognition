# Automatic-Number-plate-Recognition
ANPR using yolov3 and OpenCV

I have trained a model using yolov3 which is cloned from https://github.com/AlexeyAB/darknet
I have changed certain parameters in neural nerwork to customise it for number plate detection.You can see .cfg file above for Deep Neural network structure.Then I have prepared certain files reqiured for training in traintxtprep.ipynb.

You have to follow anprTraining.ipynb for model training.Also change makefile (in darknet folder) parameters such as CUDNN=1,CV=1,HALFCUDD=1 before training.Arrange training images and it labels' file as given in notebook.Use LabelImg tool for labeling images.

Use the training weights you got in training for number plate detection.Then convert number plate into grayscale image and remove noise using cv2 functions as given in anpd.py.Use tesseract to retrieve data from image.

For more details visit  https://github.com/AlexeyAB/darknet
