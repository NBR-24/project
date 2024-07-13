from django.db import models

# Create your models here.
from django.db import models

class Student(models.Model):
    name = models.CharField(max_length=100)
    student_id = models.CharField(max_length=50, unique=True)
    face_encoding = models.BinaryField()
    has_paid = models.BooleanField(default=False)

    def _str_(self):
        return self.name
    