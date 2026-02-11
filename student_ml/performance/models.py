from django.db import models
from django.core.exceptions import ValidationError
from django.core.validators import MinValueValidator, MaxValueValidator

class StudentPerformance(models.Model):
    hours_studied = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(16)]
    )
    sleep_hours = models.IntegerField(
        validators=[MinValueValidator(4), MaxValueValidator(12)]
    )
    previous_scores = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    extracurricular = models.BooleanField(default=False)
    sample_papers = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(20)]
    )
    performance_index = models.FloatField()

    def clean(self):
        super().clean()
        
        # Total daily hours cannot exceed 24
        total_hours = self.hours_studied + self.sleep_hours
        if total_hours > 24:
            raise ValidationError(
                f"Total hours studied ({self.hours_studied}) and sleep ({self.sleep_hours}) cannot exceed 24 hours in a day."
            )
        
        # If student has extracurricular activities, they need reasonable time for them
        if self.extracurricular and total_hours > 20:
            raise ValidationError(
                "Students with extracurricular activities cannot study/sleep more than 20 hours combined."
            )
        
        # Students need minimum sleep for good performance
        if self.sleep_hours < 6 and self.hours_studied > 12:
            raise ValidationError(
                "Students studying more than 12 hours need at least 6 hours of sleep for realistic performance."
            )

    def __str__(self):
        return f"Student ~ Index: {self.performance_index}"
