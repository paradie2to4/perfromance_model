from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import numpy as np
import joblib
from .serializers import StudentPerformanceSerializer, StudentPerformanceAnalysisSerializer
from .feature_engineering import pattern_classifier

model = joblib.load('performance/model.pkl')


@api_view(['POST'])
def predict_performance_api(request):
    """
    API endpoint for predicting student performance with comprehensive validation and pattern classification.
    
    Classifies student patterns into:
    - time_paradox: When hours exceed 24 (prediction blocked)
    - stress: When study hours are excessive with insufficient rest
    - burnout_risk: Heavy study + activities + minimal sleep
    - overachiever: Balanced high-performing pattern
    - normal: Healthy balanced approach
    
    Only proceeds with prediction for non-paradox patterns.
    """
    serializer = StudentPerformanceSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(
            {
                'error': 'Validation failed',
                'details': serializer.errors,
                'message': 'Please provide realistic student data. You cannot study 24 hours and expect good results, or sleep 24 hours and have time for extracurriculars!'
            },
            status=status.HTTP_400_BAD_REQUEST
        )
    
    data = serializer.validated_data
    
    # Check if prediction should be made based on pattern classification
    if not data.get('should_predict', True):
        return Response({
            'error': 'Prediction not recommended',
            'study_pattern': data['study_pattern'],
            'pattern_message': data['pattern_message'],
            'pattern_recommendation': data['pattern_recommendation'],
            'pattern_severity': data['pattern_severity'],
            'message': f"Cannot predict performance for '{data['study_pattern']}' pattern. Please adjust your study habits.",
            'input_data': {
                'hours_studied': data['hours_studied'],
                'sleep_hours': data['sleep_hours'],
                'previous_scores': data['previous_scores'],
                'extracurricular': data['extracurricular'],
                'sample_papers': data['sample_papers']
            }
        }, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
    
    features = np.array([[
        data['hours_studied'],
        data['previous_scores'],
        1 if data['extracurricular'] else 0,
        data['sleep_hours'],
        data['sample_papers'],
    ]])
    
    prediction = model.predict(features)[0]
    
    return Response({
        'predicted_performance_index': round(float(prediction), 2),
        'study_pattern': data['study_pattern'],
        'pattern_message': data['pattern_message'],
        'pattern_recommendation': data['pattern_recommendation'],
        'pattern_severity': data['pattern_severity'],
        'prediction_made': True,
        'input_data': {
            'hours_studied': data['hours_studied'],
            'sleep_hours': data['sleep_hours'],
            'previous_scores': data['previous_scores'],
            'extracurricular': data['extracurricular'],
            'sample_papers': data['sample_papers']
        }
    })


@api_view(['POST'])
def analyze_student_pattern(request):
    """
    Detailed analysis endpoint with engineered features and pattern classification
    """
    serializer = StudentPerformanceAnalysisSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(
            {
                'error': 'Validation failed',
                'details': serializer.errors
            },
            status=status.HTTP_400_BAD_REQUEST
        )
    
    data = serializer.validated_data
    
    return Response({
        'study_pattern': data['study_pattern'],
        'pattern_message': data['pattern_message'],
        'pattern_recommendation': data['pattern_recommendation'],
        'pattern_severity': data['pattern_severity'],
        'should_predict': data['should_predict'],
        'engineered_features': {
            'total_activity_hours': data['total_activity_hours'],
            'study_sleep_ratio': data['study_sleep_ratio'],
            'free_hours': data['free_hours'],
            'study_intensity': data['study_intensity'],
            'sleep_quality_score': data['sleep_quality_score'],
            'burnout_risk_score': data['burnout_risk_score'],
            'performance_potential': data['performance_potential']
        },
        'input_data': {
            'hours_studied': data['hours_studied'],
            'sleep_hours': data['sleep_hours'],
            'previous_scores': data['previous_scores'],
            'extracurricular': data['extracurricular'],
            'sample_papers': data['sample_papers']
        }
    })


@api_view(['GET'])
def validation_rules(request):
    """
    Returns the validation rules and pattern classifications for student data
    """
    return Response({
        'validation_rules': {
            'hours_studied': {
                'min': 0,
                'max': 16,
                'description': 'Maximum realistic study hours per day'
            },
            'sleep_hours': {
                'min': 4,
                'max': 12,
                'description': 'Healthy sleep range for students'
            },
            'previous_scores': {
                'min': 0,
                'max': 100,
                'description': 'Percentage score range'
            },
            'sample_papers': {
                'min': 0,
                'max': 20,
                'description': 'Maximum realistic practice papers'
            },
            'business_rules': [
                'Total hours (study + sleep) cannot exceed 24 hours per day',
                'Students with extracurricular activities: max 20 hours combined study/sleep',
                'Students studying >12 hours need minimum 6 hours sleep for realistic performance'
            ]
        },
        'pattern_classifications': {
            'time_paradox': {
                'description': 'When total hours exceed 24 - physically impossible',
                'prediction_allowed': False,
                'severity': 'critical',
                'example': 'Study 16h + Sleep 10h = 26h total'
            },
            'stress': {
                'description': 'Excessive studying with insufficient rest',
                'prediction_allowed': True,
                'severity': 'medium',
                'example': 'Study 15h + Sleep 5h = 20h total'
            },
            'burnout_risk': {
                'description': 'Heavy study load with activities and minimal sleep',
                'prediction_allowed': True,
                'severity': 'high',
                'example': 'Study 12h + Sleep 5h + Activities = High risk'
            },
            'overachiever': {
                'description': 'Balanced high-performing pattern',
                'prediction_allowed': True,
                'severity': 'low',
                'example': 'Study 10h + Sleep 8h + Activities + Practice'
            },
            'normal': {
                'description': 'Healthy balanced approach',
                'prediction_allowed': True,
                'severity': 'low',
                'example': 'Study 6h + Sleep 8h = 14h total'
            }
        }
    })


@api_view(['GET'])
def pattern_info(request):
    """
    Returns information about available patterns and their characteristics
    """
    return Response({
        'pattern_info': {
            'hours_studied': {
                'min': 0,
                'max': 16,
                'description': 'Maximum realistic study hours per day'
            },
            'sleep_hours': {
                'min': 4,
                'max': 12,
                'description': 'Healthy sleep range for students'
            },
            'previous_scores': {
                'min': 0,
                'max': 100,
                'description': 'Percentage score range'
            },
            'sample_papers': {
                'min': 0,
                'max': 20,
                'description': 'Maximum realistic practice papers'
            },
            'business_rules': [
                'Total hours (study + sleep) cannot exceed 24 hours per day',
                'Students with extracurricular activities: max 20 hours combined study/sleep',
                'Students studying >12 hours need minimum 6 hours sleep for realistic performance'
            ]
        }
    })
