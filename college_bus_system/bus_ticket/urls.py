from django.urls import path
from .views import confirm_payment

urlpatterns = [
    path('', index, name='index'),
    path('register_student/', register_student, name='register_student'),
    path('confirm_payment/', confirm_payment, name='confirm_payment'),
]