from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import time
import psycopg2
from datetime import datetime, timedelta
from collections import defaultdict

# Database connection
conn = psycopg2.connect(
    database="bus",
    user="postgres",
    password="aslam123",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Constants
DETECTION_LIMIT = 100  # Number of detections required
TIME_LIMIT_HOURS = 3  # Time limit in hours

# Variables
face_counters = defaultdict(int)
start_times = defaultdict(lambda: time.time())
last_attendance_time = defaultdict(lambda: datetime.min)

# Initialize video capture and face cascade
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load the trained models and data
with open('data/reg_no.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Main loop for face detection and recognition
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    cv2.putText(frame, "press 'Q' to exit", (30, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (50, 50, 255), 1)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        try:
            output = knn.predict(resized_img)
            detected_admno = output[0]

            cv2.putText(frame, str(detected_admno), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
            

            face_counters[detected_admno] += 1

            if face_counters[detected_admno] >= DETECTION_LIMIT:
                current_time = datetime.now()
                if current_time - last_attendance_time[detected_admno] >= timedelta(hours=TIME_LIMIT_HOURS):
                    # Insert attendance record into PostgreSQL
                    time_stamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    sql = "UPDATE data SET time_stamp = %s WHERE adm_no = %s"
                    val = (time_stamp, detected_admno)
                    cursor.execute(sql, val)
                    conn.commit()

                    print(f"Ticket taken for {detected_admno} at {time_stamp}")

                    # Reset counter and update the last attendance time for the detected name
                    face_counters[detected_admno] = 0
                    last_attendance_time[detected_admno] = current_time
                else:
                    # Reset counter
                    face_counters[detected_admno] = 0
        except Exception as e:
            print(f"Error during prediction: {e}")

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release video capture and close all windows
video.release()
cv2.destroyAllWindows()

# Close database connection
cursor.close()
conn.close()