from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse
from .models import Student
import numpy as np
import cv2
import face_recognition

def index(request):
    return render(request, 'bus_ticket/index.html')

def register_student(request):
    if request.method == 'POST':
        name = request.POST['name']
        student_id = request.POST['student_id']
        face_encoding = capture_face()

        if face_encoding is not None:
            Student.objects.create(name=name, student_id=student_id, face_encoding=face_encoding.tobytes())
            return JsonResponse({'status': 'success'})
        return JsonResponse({'status': 'fail'})

def confirm_payment(request):
    recognized_student = recognize_face()
    if recognized_student:
        student = Student.objects.get(student_id=recognized_student)
        if student.has_paid:
            return JsonResponse({'status': 'access_granted', 'name': student.name})
        return JsonResponse({'status': 'payment_required', 'name': student.name})
    return JsonResponse({'status': 'no_match'})

def capture_face():
    video_capture = cv2.VideoCapture(0)
    face_encoding = None

    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            face_encoding = face_encodings[0]
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return face_encoding

def recognize_face():
    video_capture = cv2.VideoCapture(0)
    recognized_student = None

    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        students = Student.objects.all()
        known_face_encodings = [np.frombuffer(student.face_encoding) for student in students]
        known_face_ids = [student.student_id for student in students]

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                recognized_student = known_face_ids[best_match_index]
                break

        if recognized_student:
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return recognized_student
