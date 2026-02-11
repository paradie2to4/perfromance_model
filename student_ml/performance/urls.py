from django.urls import path
from . import views
from . import api_views

urlpatterns = [
    path('train/', views.train_model_page, name='train_model_page'),
    path('predict/', views.predict_performance, name='predict_performance'),
    path('predict-ui/', views.predict_model_page, name='predict_model_page'),
    path('api/predict/', api_views.predict_performance_api, name='predict_performance_api'),
    path('api/analyze/', api_views.analyze_student_pattern, name='analyze_student_pattern'),
    path('api/validation-rules/', api_views.validation_rules, name='validation_rules'),
    path('api/pattern-info/', api_views.pattern_info, name='pattern_info'),
]