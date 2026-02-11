# Student Performance ML Model

A Django-based machine learning application for predicting student performance and analyzing educational data with bias detection and mitigation.

## Features

- **Student Performance Prediction**: ML models to predict student performance based on various features
- **Bias Detection & Mitigation**: Tools to identify and reduce bias in predictions
- **Feature Engineering**: Automated feature engineering for improved model performance
- **REST API**: Django REST Framework API for model predictions
- **Admin Dashboard**: Django admin interface for data management
- **Data Management**: SQLite database for storing student performance records

## Project Structure

```
student_ml/
├── manage.py                          # Django management script
├── db.sqlite3                         # SQLite database
├── Student_Performance.csv            # Dataset
├── student_ml/                        # Django project settings
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
└── performance/                       # Main Django app
    ├── models.py                      # Database models
    ├── views.py                       # View functions
    ├── api_views.py                   # REST API views
    ├── serializers.py                 # DRF serializers
    ├── urls.py                        # URL routing
    ├── train_model.py                 # Model training logic
    ├── train_model_unbiased.py        # Unbiased model training
    ├── bias_utils.py                  # Bias detection utilities
    ├── feature_engineering.py         # Feature engineering functions
    ├── load_data.py                   # Data loading utilities
    ├── admin.py                       # Admin configuration
    └── templates/
        └── performance/               # HTML templates
            ├── train.html
            └── predict.html
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/paradie2to4/perfromance_model.git
   cd student_ml
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply migrations**
   ```bash
   python manage.py migrate
   ```

5. **Create a superuser** (for admin access)
   ```bash
   python manage.py createsuperuser
   ```

## Usage

### Running the Development Server

```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`

### Admin Interface

Access the admin dashboard at `http://127.0.0.1:8000/admin/` with your superuser credentials.

### Training Models

The project includes two training approaches:

- **Standard Model**: `train_model.py` - Standard ML model training
- **Unbiased Model**: `train_model_unbiased.py` - Model training with bias mitigation

To train a model, use the web interface or API endpoints.

### API Endpoints

- `GET /api/predictions/` - List all predictions
- `POST /api/predict/` - Create a new prediction
- `GET /api/student-performance/` - List student performance records

## Technologies Used

- **Django**: Web framework
- **Django REST Framework**: RESTful API
- **Scikit-learn**: Machine learning
- **Pandas**: Data analysis
- **SQLite**: Database

## Features

### ML Model Training
- Trains on student performance data
- Supports multiple ML algorithms
- Evaluates model performance metrics

### Bias Detection
- Analyzes predictions for bias
- Identifies disparate impact
- Provides mitigation strategies

### Feature Engineering
- Automatic feature transformation
- Feature scaling and normalization
- Categorical encoding

## Configuration

Update settings in `student_ml/settings.py`:
- Database configuration
- Installed apps
- Middleware
- Static files location

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- **GitHub**: [paradie2to4](https://github.com/paradie2to4)

## Support

For support, please open an issue on the GitHub repository.

---

**Note**: This project is designed for educational purposes to demonstrate ML model training, deployment, and bias detection in educational data.
