from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import os

# Create your views here.
import joblib
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import StudentPerformanceSerializer

model = joblib.load('performance/model.pkl')

@api_view(['POST'])
def predict_performance(request):
    serializer = StudentPerformanceSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(
            serializer.errors,
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
            'message': 'Cannot predict performance for this pattern. Please adjust your study habits.'
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
        'prediction_made': True
    })

def train_model_page(request):
    if request.method == 'POST':
        # Here you would add your model training logic
        # For demonstration, just simulate training
        # Save a message to show training is done
        context = {'message': 'Training completed successfully!'}
        return render(request, 'performance/train.html', context)
    return render(request, 'performance/train.html')

def predict_model_page(request):
    if request.method == 'POST':
        try:
            # Create data dict for serializer validation
            data = {
                'hours_studied': float(request.POST.get('hours_studied')),
                'previous_scores': float(request.POST.get('previous_scores')),
                'extracurricular': request.POST.get('extracurricular') == 'on',
                'sleep_hours': float(request.POST.get('sleep_hours')),
                'sample_papers': float(request.POST.get('sample_papers'))
            }
            
            # Validate data using serializer
            serializer = StudentPerformanceSerializer(data=data)
            if not serializer.is_valid():
                context = {'error': serializer.errors}
                return render(request, 'performance/predict.html', context)
            
            validated_data = serializer.validated_data
            
            # Check if prediction should be made based on pattern classification
            if not validated_data.get('should_predict', True):
                context = {
                    'pattern_error': True,
                    'study_pattern': validated_data['study_pattern'],
                    'pattern_message': validated_data['pattern_message'],
                    'pattern_recommendation': validated_data['pattern_recommendation'],
                    'pattern_severity': validated_data['pattern_severity']
                }
                return render(request, 'performance/predict.html', context)
            
            features = np.array([[
                validated_data['hours_studied'],
                validated_data['previous_scores'],
                1 if validated_data['extracurricular'] else 0,
                validated_data['sleep_hours'],
                validated_data['sample_papers']
            ]])
            prediction = model.predict(features)[0]
            
            context = {
                'prediction': round(float(prediction), 2),
                'study_pattern': validated_data['study_pattern'],
                'pattern_message': validated_data['pattern_message'],
                'pattern_recommendation': validated_data['pattern_recommendation'],
                'pattern_severity': validated_data['pattern_severity']
            }
        except Exception as e:
            context = {'error': str(e)}
        return render(request, 'performance/predict.html', context)
    return render(request, 'performance/predict.html')