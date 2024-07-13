import cv2
import pickle
import numpy as np
import os
import sys
import time
import psycopg2

# Database connection parameters
db_params = {
    'dbname': 'bus',
    'user': 'postgres',
    'password': 'aslam123',
    'host': 'localhost',
    'port': '5432'
}

def connect_to_db():
    try:
        conn = psycopg2.connect(**db_params)
        print("Connected to the database successfully!")
        return conn
    except Exception as e:
        print(f"Unable to connect to the database: {e}")
        return None

def insert_data(conn, name, adm_no, amount):
    try:
        with conn.cursor() as cursor:
            insert_query = """
            INSERT INTO data (name, adm_no, amount)
            VALUES (%s, %s, %s);
            """
            cursor.execute(insert_query, (name, adm_no, amount))
            conn.commit()
            print("Data inserted successfully!")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting data: {e}")

def check_reg_no_exists(conn, reg_no):
    with conn.cursor() as cur:
        query = """
        SELECT COUNT(*)
        FROM data
        WHERE adm_no = %s;
        """
        cur.execute(query, (reg_no,))
        count = cur.fetchone()[0]
    return count > 0

# Video capturing using webcam
video = cv2.VideoCapture(0)

# Adding face detection algorithm (predefined)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
faces_data = []
i = 0

if not os.path.exists('data'):
    os.makedirs('data')

name = input("Enter your Name: ")
reg_no = input("Enter your admission no: ")
stop = input("\nAvailable services from college \n\n Cherpulassery [code : CPY] : 3Rs/trip\n Pattambi [code : PTB] : 5Rs/trip \n Shornur [code : SHR] : 10Rs/trip\n \n Enter your Place code : " )
stop = stop.upper()

while True:
    if stop == "CPY":
        amount = 3
        break
    elif stop == "PTB":
        amount = 5
        break
    elif stop == "SHR":
        amount = 10
        break
    else:
        print("Enter a correct input: ")  
        stop = input().upper()

conn = connect_to_db()
if conn:
    exists = check_reg_no_exists(conn, reg_no)
    if exists:
        print(f"Admission number {reg_no} exists in the database. EXITING....")
        time.sleep(2)
        sys.exit()
    else:
        while True:
            ret, frame = video.read()
            greyscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = facedetect.detectMultiScale(greyscale, 1.3, 5)
            
            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w, :]
                resize_img = cv2.resize(crop_img, (50, 50))
                if len(faces_data) < 100 and i % 10 == 0:
                    faces_data.append(resize_img)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

            i += 1
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.putText(frame, "Recording face, please wait", (0, 400), cv2.FONT_HERSHEY_TRIPLEX, 1, (50, 50, 255), 1)
            cv2.putText(frame, "until count becomes 100", (0, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (50, 50, 255), 1)
            cv2.putText(frame, "press 'Q' to exit", (30, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (50, 50, 255), 1)
            cv2.imshow("Frame", frame)

            k = cv2.waitKey(1)
            if k == ord('q'):
                break

            if len(faces_data) == 100:
                print("Image captured successfully.")
                insert_data(conn, name, reg_no, amount)
                break

        video.release()
        cv2.destroyAllWindows()

        if len(faces_data) == 100:
            faces_data = np.asarray(faces_data)
            faces_data = faces_data.reshape(len(faces_data), -1)

            if 'reg_no.pkl' not in os.listdir('data/'):
                reg_no_list = [reg_no] * 100
                with open('data/reg_no.pkl', 'wb') as f:
                    pickle.dump(reg_no_list, f)
            else:
                with open('data/reg_no.pkl', 'rb') as f:
                    reg_no_list = pickle.load(f)
                reg_no_list.extend([reg_no] * 100)
                with open('data/reg_no.pkl', 'wb') as f:
                    pickle.dump(reg_no_list, f)

            if 'faces_data.pkl' not in os.listdir('data/'):
                with open('data/faces_data.pkl', 'wb') as f:
                    pickle.dump(faces_data, f)
            else:
                with open('data/faces_data.pkl', 'rb') as f:
                    faces = pickle.load(f)
                faces = np.append(faces, faces_data, axis=0)
                with open('data/faces_data.pkl', 'wb') as f:
                    pickle.dump(faces, f)
        else:
            print("No faces were captured. Please try again.")
    conn.close()
else:
    print("Failed to connect to the database. Exiting...")