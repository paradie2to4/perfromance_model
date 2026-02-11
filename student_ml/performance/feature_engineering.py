"""
Feature Engineering module for Student Performance Prediction
Classifies student study patterns into meaningful categories
"""

from enum import Enum
from typing import Dict, Tuple, Any


class StudyPattern(Enum):
    TIME_PARADOX = "time_paradox"
    STRESS = "stress" 
    NORMAL = "normal"
    OVERACHIEVER = "overachiever"
    BURNOUT_RISK = "burnout_risk"


class StudentPatternClassifier:
    """
    Classifies student study patterns based on hours and lifestyle factors
    """
    
    def __init__(self):
        self.classification_rules = {
            'time_paradox': {
                'condition': lambda data: data['hours_studied'] + data['sleep_hours'] > 24,
                'message': "Time paradox detected! You cannot study and sleep more than 24 hours in a day.",
                'recommendation': "Please review your time allocation - physics doesn't allow this!"
            },
            'stress': {
                'condition': lambda data: (
                    data['hours_studied'] > 14 or 
                    (data['hours_studied'] > 10 and data['sleep_hours'] < 6)
                ),
                'message': "High stress pattern detected - excessive studying with insufficient rest.",
                'recommendation': "Consider balancing study time with proper rest for better performance."
            },
            'burnout_risk': {
                'condition': lambda data: (
                    data['hours_studied'] >= 12 and 
                    data['sleep_hours'] <= 5 and
                    data['extracurricular']
                ),
                'message': "Burnout risk pattern - heavy study load with activities and minimal sleep.",
                'recommendation': "Prioritize sleep and consider reducing extracurricular commitments."
            },
            'overachiever': {
                'condition': lambda data: (
                    8 <= data['hours_studied'] <= 12 and
                    7 <= data['sleep_hours'] <= 9 and
                    data['extracurricular'] and
                    data['sample_papers'] >= 5
                ),
                'message': "Balanced overachiever pattern - excellent time management.",
                'recommendation': "Great balance! Maintain this healthy routine."
            },
            'normal': {
                'condition': lambda data: (
                    4 <= data['hours_studied'] <= 10 and
                    6 <= data['sleep_hours'] <= 10
                ),
                'message': "Normal study pattern - balanced approach.",
                'recommendation': "Good balance between study and rest."
            }
        }
    
    def classify_pattern(self, data: Dict[str, Any]) -> Tuple[StudyPattern, Dict[str, Any]]:
        """
        Classifies student study pattern and returns classification details
        
        Args:
            data: Dictionary containing student data
            
        Returns:
            Tuple of (StudyPattern enum, classification_details)
        """
        
        # Check time paradox first (most critical)
        if data['hours_studied'] + data['sleep_hours'] > 24:
            return StudyPattern.TIME_PARADOX, {
                'pattern': 'time_paradox',
                'message': self.classification_rules['time_paradox']['message'],
                'recommendation': self.classification_rules['time_paradox']['recommendation'],
                'should_predict': False,
                'severity': 'critical'
            }
        
        # Check burnout risk
        if (data['hours_studied'] >= 12 and 
            data['sleep_hours'] <= 5 and
            data['extracurricular']):
            return StudyPattern.BURNOUT_RISK, {
                'pattern': 'burnout_risk',
                'message': self.classification_rules['burnout_risk']['message'],
                'recommendation': self.classification_rules['burnout_risk']['recommendation'],
                'should_predict': True,
                'severity': 'high'
            }
        
        # Check stress pattern
        if (data['hours_studied'] > 14 or 
            (data['hours_studied'] > 10 and data['sleep_hours'] < 6)):
            return StudyPattern.STRESS, {
                'pattern': 'stress',
                'message': self.classification_rules['stress']['message'],
                'recommendation': self.classification_rules['stress']['recommendation'],
                'should_predict': True,
                'severity': 'medium'
            }
        
        # Check overachiever pattern
        if (8 <= data['hours_studied'] <= 12 and
            7 <= data['sleep_hours'] <= 9 and
            data['extracurricular'] and
            data['sample_papers'] >= 5):
            return StudyPattern.OVERACHIEVER, {
                'pattern': 'overachiever',
                'message': self.classification_rules['overachiever']['message'],
                'recommendation': self.classification_rules['overachiever']['recommendation'],
                'should_predict': True,
                'severity': 'low'
            }
        
        # Default to normal
        return StudyPattern.NORMAL, {
            'pattern': 'normal',
            'message': self.classification_rules['normal']['message'],
            'recommendation': self.classification_rules['normal']['recommendation'],
            'should_predict': True,
            'severity': 'low'
        }
    
    def get_pattern_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extracts engineered features from the input data
        
        Args:
            data: Dictionary containing student data
            
        Returns:
            Dictionary of engineered features
        """
        features = {}
        
        # Time-based features
        features['total_activity_hours'] = data['hours_studied'] + data['sleep_hours']
        features['study_sleep_ratio'] = data['hours_studied'] / max(data['sleep_hours'], 1)
        features['free_hours'] = 24 - features['total_activity_hours']
        
        # Intensity features
        features['study_intensity'] = min(data['hours_studied'] / 12, 1.0)  # Normalized to 12h max
        features['sleep_quality_score'] = min(data['sleep_hours'] / 8, 1.0)  # Normalized to 8h optimal
        
        # Lifestyle balance features
        features['has_extracurricular'] = 1 if data['extracurricular'] else 0
        features['practice_intensity'] = min(data['sample_papers'] / 10, 1.0)  # Normalized to 10 papers
        
        # Risk scoring
        features['burnout_risk_score'] = self._calculate_burnout_risk(data)
        features['performance_potential'] = self._calculate_performance_potential(data)
        
        return features
    
    def _calculate_burnout_risk(self, data: Dict[str, Any]) -> float:
        """Calculate burnout risk score (0-1, higher is worse)"""
        risk = 0.0
        
        # High study hours increase risk
        if data['hours_studied'] > 12:
            risk += 0.4
        elif data['hours_studied'] > 10:
            risk += 0.2
        
        # Low sleep increases risk
        if data['sleep_hours'] < 5:
            risk += 0.4
        elif data['sleep_hours'] < 6:
            risk += 0.2
        
        # Extracurricular + high load increases risk
        if data['extracurricular'] and data['hours_studied'] > 10:
            risk += 0.2
        
        return min(risk, 1.0)
    
    def _calculate_performance_potential(self, data: Dict[str, Any]) -> float:
        """Calculate performance potential score (0-1, higher is better)"""
        potential = 0.5  # Base score
        
        # Optimal study hours
        if 6 <= data['hours_studied'] <= 10:
            potential += 0.2
        elif data['hours_studied'] > 14:
            potential -= 0.2
        
        # Good sleep
        if 7 <= data['sleep_hours'] <= 9:
            potential += 0.2
        elif data['sleep_hours'] < 6:
            potential -= 0.1
        
        # Practice papers
        if data['sample_papers'] >= 5:
            potential += 0.1
        
        return min(max(potential, 0.0), 1.0)


# Global classifier instance
pattern_classifier = StudentPatternClassifier()
