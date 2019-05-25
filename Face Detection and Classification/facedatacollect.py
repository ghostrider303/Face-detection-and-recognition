import cv2
import numpy as np
file_name=input("Enter name: ")
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
skip=0
face_data=[]
dataset_path="./data/"
while True:
	ret,frame=cap.read()
	if ret==False:
		continue
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(frame,1.3,5)
	faces=sorted(faces, key= lambda f:f[2]*f[3] )
	for face in faces[-1:]:
		x,y,w,h=face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
		#crop required paart
		offset=10
		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))
		skip+=1
		if(skip%10==0):
			face_data.append(face_section)
			print(len(face_data))
		cv2.imshow("Frame",frame)
		cv2.imshow("Face section",face_section)
	key=cv2.waitKey(1) & 0xFF
	if key==ord("q"):
		break
#conver face list array nto nupy aray
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
#save this into file system
np.save(dataset_path+file_name+".npy",face_data)
print("Data succesfully saved at",dataset_path+file_name+".npy")

cap.release()
cv2.destroyAllWindows()
