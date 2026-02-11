from rest_framework import serializers
from django.core.exceptions import ValidationError
from .models import StudentPerformance
from .feature_engineering import pattern_classifier


class StudentPerformanceSerializer(serializers.ModelSerializer):
    # Add read-only fields for classification results
    study_pattern = serializers.CharField(read_only=True)
    pattern_message = serializers.CharField(read_only=True)
    pattern_recommendation = serializers.CharField(read_only=True)
    should_predict = serializers.BooleanField(read_only=True)
    pattern_severity = serializers.CharField(read_only=True)
    
    class Meta:
        model = StudentPerformance
        fields = [
            'hours_studied', 'previous_scores', 'extracurricular', 'sleep_hours', 'sample_papers',
            'study_pattern', 'pattern_message', 'pattern_recommendation', 'should_predict', 'pattern_severity'
        ]
    
    def validate_hours_studied(self, value):
        if value < 0 or value > 16:
            raise serializers.ValidationError("Hours studied must be between 0 and 16 hours per day.")
        return value
    
    def validate_sleep_hours(self, value):
        if value < 4 or value > 12:
            raise serializers.ValidationError("Sleep hours must be between 4 and 12 hours per day.")
        return value
    
    def validate_previous_scores(self, value):
        if value < 0 or value > 100:
            raise serializers.ValidationError("Previous scores must be between 0 and 100.")
        return value
    
    def validate_sample_papers(self, value):
        if value < 0 or value > 20:
            raise serializers.ValidationError("Sample papers practiced must be between 0 and 20.")
        return value
    
    def validate(self, data):
        hours_studied = data.get('hours_studied', 0)
        sleep_hours = data.get('sleep_hours', 0)
        extracurricular = data.get('extracurricular', False)
        
        # Total daily hours cannot exceed 24
        total_hours = hours_studied + sleep_hours
        if total_hours > 24:
            raise serializers.ValidationError(
                f"Total hours studied ({hours_studied}) and sleep ({sleep_hours}) cannot exceed 24 hours in a day."
            )
        
        # If student has extracurricular activities, they need reasonable time for them
        if extracurricular and total_hours > 20:
            raise serializers.ValidationError(
                "Students with extracurricular activities cannot study/sleep more than 20 hours combined."
            )
        
        # Students need minimum sleep for good performance
        if sleep_hours < 6 and hours_studied > 12:
            raise serializers.ValidationError(
                "Students studying more than 12 hours need at least 6 hours of sleep for realistic performance."
            )
        
        # Add classification to validated data
        pattern, classification_details = pattern_classifier.classify_pattern(data)
        data.update({
            'study_pattern': pattern.value,
            'pattern_message': classification_details['message'],
            'pattern_recommendation': classification_details['recommendation'],
            'should_predict': classification_details['should_predict'],
            'pattern_severity': classification_details['severity']
        })
        
        return data


class StudentPerformanceAnalysisSerializer(serializers.Serializer):
    """
    Serializer for detailed analysis including engineered features
    """
    hours_studied = serializers.IntegerField(min_value=0, max_value=16)
    previous_scores = serializers.IntegerField(min_value=0, max_value=100)
    extracurricular = serializers.BooleanField()
    sleep_hours = serializers.IntegerField(min_value=4, max_value=12)
    sample_papers = serializers.IntegerField(min_value=0, max_value=20)
    
    # Classification results
    study_pattern = serializers.CharField(read_only=True)
    pattern_message = serializers.CharField(read_only=True)
    pattern_recommendation = serializers.CharField(read_only=True)
    should_predict = serializers.BooleanField(read_only=True)
    pattern_severity = serializers.CharField(read_only=True)
    
    # Engineered features
    total_activity_hours = serializers.FloatField(read_only=True)
    study_sleep_ratio = serializers.FloatField(read_only=True)
    free_hours = serializers.FloatField(read_only=True)
    study_intensity = serializers.FloatField(read_only=True)
    sleep_quality_score = serializers.FloatField(read_only=True)
    burnout_risk_score = serializers.FloatField(read_only=True)
    performance_potential = serializers.FloatField(read_only=True)
    
    def validate(self, data):
        # Use the same validation as the main serializer
        main_serializer = StudentPerformanceSerializer(data=data)
        if not main_serializer.is_valid():
            raise serializers.ValidationError(main_serializer.errors)
        
        # Add classification and engineered features
        pattern, classification_details = pattern_classifier.classify_pattern(data)
        features = pattern_classifier.get_pattern_features(data)
        
        data.update({
            'study_pattern': pattern.value,
            'pattern_message': classification_details['message'],
            'pattern_recommendation': classification_details['recommendation'],
            'should_predict': classification_details['should_predict'],
            'pattern_severity': classification_details['severity'],
            **features
        })
        
        return data
