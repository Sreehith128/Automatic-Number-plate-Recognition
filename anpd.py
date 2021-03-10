import cv2
import numpy as np
import pytesseract
import re

states=['AP','AR','AS','BR','CG','GA','GJ','HR','HP','JK','JH','KA','KL','MP','MH','MN','ML',
		'MZ','NL','OR','PB','RJ','SK','TN','TS','TR','UK','UP','WB','AN','CH','DH','DD','DL',
		'LD','PY'
		]

net = cv2.dnn.readNet("yolov4_train_final.weights", "yolov4_train.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

cam=cv2.VideoCapture('test.mp4')

#class_names=['number plate']
numbers=[]
while True:
	s,frame=cam.read()
	if not s:break
	#
	#frame=cv2.imread('test.jpg')
	frame = cv2.resize(frame, (1020,600), interpolation = cv2.INTER_AREA)
	#filtered=None

	classes,scores,boxes=model.detect(frame,0.3,0.5)
	for (classid, score, box) in zip(classes, scores, boxes):
		#label = "%s : %f" % (class_names[classid[0]], score)
		cv2.rectangle(frame, box, (0,255,0), 2)
		#cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
		if len(box)==4:
			numPlate=frame[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
			
			gray=cv2.cvtColor(numPlate,cv2.COLOR_BGR2GRAY)
			ret,thresh=cv2.threshold(gray,140,255,cv2.THRESH_BINARY)
			#thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
			filtered=cv2.GaussianBlur(thresh,(5,5),0)

			number = pytesseract.image_to_string(filtered)[:-2]
			#print(number,'not')
			number=number.strip().replace(' ','')
			number=re.sub('{*,*}*-*_*\?*','',number)
			for i in number.split('\n'):
				#if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]{0,3}[0-9]{4}$',i):
					#print(i,'end')
				if i[:2] in states:
					cv2.putText(frame,i,(box[0],box[1]-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
					if i not in numbers:
						numbers.append(i)

	cv2.imshow('number plate detection',frame)
	#cv2.imshow('plate',filtered)

	if cv2.waitKey(1) & 0xff == ord('q'):
		break


cv2.destroyAllWindows()
print(numbers)

