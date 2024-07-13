import cv2
import pickle
import numpy as np
import os


#video capturing using webcam
video=cv2.VideoCapture(0)



#adding face detection alogorithm 
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
faces_data=[]
i=0


name=input("Enter your Name : ")
reg_no=input("Enter your Reg no : ")




#looping until we get a image
while True:
    ret,frame=video.read()

    #converting to greyscale for better understanding
    greyscale=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces=facedetect.detectMultiScale(greyscale, 1.3,5)

    #drawing rectangle arounf detected faces and cropping
    for (x,y,w,h) in faces:

       crop_img= frame[y:y+h,  x:x+w,  :]
       resize_img=cv2.resize(crop_img,(50,50))
       

       if len(faces_data)<=100 and i%10==0:
        faces_data.append(resize_img)

       
       i=i+1
       cv2.putText(frame, str(len(faces_data)),(50,50),cv2.FONT_HERSHEY_COMPLEX,1 ,(50,50,255),1)

       cv2.rectangle(frame,(x,y),(x+w ,y+h),(50,50,255), 1)

    #showing the webcam
    cv2.imshow("Frame",frame)


    #quit cam while pressing q
    k=cv2.waitKey(1)
    if k==ord('q') or len(faces_data)==100:
        break


video.release()
cv2.destroyAllWindows()






faces_data=np.asarray(faces_data)
faces_data=faces_data.reshape(100,-1)






if 'names.pkl' not in os.listdir('data/'):
    names=[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names=pickle.load(f)
    names=names+[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)