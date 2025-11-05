from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'user'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    blood_group = db.Column(db.String(5), nullable=False)
    address = db.Column(db.Text, nullable=False)
    profile_pic = db.Column(db.String(200), default='default.jpg')
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    health_parameters = db.relationship('HealthParameter', backref='user', lazy=True, cascade='all, delete-orphan')
    reports = db.relationship('MedicalReport', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.email}>'

class HealthParameter(db.Model):
    __tablename__ = 'health_parameter'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    heart_rate = db.Column(db.Integer)
    systolic_bp = db.Column(db.Integer)  # Upper BP value
    diastolic_bp = db.Column(db.Integer)  # Lower BP value
    spo2 = db.Column(db.Integer)  # Oxygen saturation
    temperature = db.Column(db.Float)
    notes = db.Column(db.Text)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<HealthParameter {self.id} for User {self.user_id}>'

class MedicalReport(db.Model):
    __tablename__ = 'medical_report'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    file_path = db.Column(db.String(300), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    results = db.Column(db.Text)  # Doctor's analysis or results
    status = db.Column(db.String(20), default='pending')  # pending, reviewed, completed
    
    def __repr__(self):
        return f'<MedicalReport {self.title}>'
