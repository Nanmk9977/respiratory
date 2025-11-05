from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from models import db, User, HealthParameter, MedicalReport
from config import Config
import os
from datetime import datetime
from PIL import Image
import pickle
import numpy as np
import random
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from skimage import transform
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


def gradcam(fname):
    model = models.resnet50(pretrained=True)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        ])
    image_path = fname
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    gradients = []
    activations = []
    def save_activation(module, input, output):
        activations.append(output)
    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[-1].conv3
    target_layer.register_forward_hook(save_activation)
    target_layer.register_backward_hook(save_gradient)
    output = model(input_tensor)
    pred_class = output.argmax().item()
    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img.size[0], img.size[1]))
    cam -= cam.min()
    cam /= cam.max()
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    img_np = np.array(img)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original X-ray")
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(heatmap)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("static/gradcam_output.png", bbox_inches='tight', dpi=300)
    plt.show()

with open("respiratory_rf_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    loaded_scaler = pickle.load(scaler_file)



MODEL_PATH ='lungs.h5'

resnet_model = load_model(MODEL_PATH)

resclass=['Asthma', 'Atelectasis', 'Bronchitis', 'COPD', 'Normal', 'Pleural Effusion', 'Pneumonia', 'Pulmonary Fibrosis', 'Tuberculose']

photo_size=256

def predict_img(fpath):
    
    image=cv2.imread(fpath)
    example = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(example)
    image_resized= cv2.resize(image, (256,256))
    image=np.expand_dims(image_resized,axis=0)
    pred=resnet_model.predict(image)
    output=resclass[np.argmax(pred)]
    return(output)

def load_image_from_path(filename):
    img = mpimg.imread(filename)
    imgplot = plt.imshow(img)
    plt.show()
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (photo_size, photo_size, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize extensions
    db.init_app(app)
    
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Create tables within app context
    with app.app_context():
        db.create_all()
    
    # Ensure upload directories exist
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'profiles'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'reports'), exist_ok=True)
    
    # Create default profile picture if it doesn't exist
    default_pic_path = os.path.join(app.root_path, 'static/uploads/profiles/default.jpg')
    if not os.path.exists(default_pic_path):
        # Create a simple default image
        img = Image.new('RGB', (125, 125), color='lightgray')
        img.save(default_pic_path)
    
    def allowed_file(filename, allowed_extensions):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in allowed_extensions
    
    def save_picture(form_picture):
        random_hex = os.urandom(8).hex()
        _, f_ext = os.path.splitext(form_picture.filename)
        picture_fn = random_hex + f_ext
        picture_path = os.path.join(app.root_path, 'static/uploads/profiles', picture_fn)
        
        # Resize image before saving
        output_size = (125, 125)
        i = Image.open(form_picture)
        i.thumbnail(output_size)
        i.save(picture_path)
        
        return picture_fn
    
    # Routes remain the same as before...
    @app.route('/')
    def index():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        return redirect(url_for('login'))
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        if request.method == 'POST':
            email = request.form.get('email')
            password = request.form.get('password')
            user = User.query.filter_by(email=email).first()
            
            if user and user.check_password(password):
                login_user(user)
                next_page = request.args.get('next')
                return redirect(next_page) if next_page else redirect(url_for('dashboard'))
            else:
                flash('Login failed. Check your email and password.', 'danger')
        
        return render_template('login.html')
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        
        if request.method == 'POST':
            name = request.form.get('name')
            email = request.form.get('email')
            phone = request.form.get('phone')
            gender = request.form.get('gender')
            age = request.form.get('age')
            blood_group = request.form.get('blood_group')
            address = request.form.get('address')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            if password != confirm_password:
                flash('Passwords do not match!', 'danger')
                return render_template('register.html')
            
            if User.query.filter_by(email=email).first():
                flash('Email already exists!', 'danger')
                return render_template('register.html')
            
            profile_pic = 'default.jpg'
            if 'profile_pic' in request.files:
                file = request.files['profile_pic']
                if file and file.filename != '' and allowed_file(file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
                    profile_pic = save_picture(file)
            
            user = User(
                name=name, email=email, phone=phone, gender=gender,
                age=age, blood_group=blood_group, address=address,
                profile_pic=profile_pic
            )
            user.set_password(password)
            
            db.session.add(user)
            db.session.commit()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        
        return render_template('register.html')
    
    @app.route('/dashboard')
    @login_required
    def dashboard():
        # Get recent health parameters
        recent_params = HealthParameter.query.filter_by(user_id=current_user.id)\
            .order_by(HealthParameter.recorded_at.desc()).limit(5).all()
        
        # Get recent reports
        recent_reports = MedicalReport.query.filter_by(user_id=current_user.id)\
            .order_by(MedicalReport.uploaded_at.desc()).limit(5).all()
        
        return render_template('dashboard.html', 
                             recent_params=recent_params,
                             recent_reports=recent_reports)
    
    @app.route('/profile', methods=['GET', 'POST'])
    @login_required
    def profile():
        if request.method == 'POST':
            current_user.name = request.form.get('name')
            current_user.phone = request.form.get('phone')
            current_user.gender = request.form.get('gender')
            current_user.age = request.form.get('age')
            current_user.blood_group = request.form.get('blood_group')
            current_user.address = request.form.get('address')
            
            if 'profile_pic' in request.files:
                file = request.files['profile_pic']
                if file and file.filename != '' and allowed_file(file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
                    # Delete old profile picture if it's not default
                    if current_user.profile_pic != 'default.jpg':
                        old_pic_path = os.path.join(app.root_path, 'static/uploads/profiles', current_user.profile_pic)
                        if os.path.exists(old_pic_path):
                            os.remove(old_pic_path)
                    
                    current_user.profile_pic = save_picture(file)
            
            db.session.commit()
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('profile'))
        
        return render_template('profile.html')
    
    @app.route('/health-parameters', methods=['GET', 'POST'])
    @login_required
    def health_parameters():
        if request.method == 'POST':
            heart_rate = request.form.get('heart_rate')
            systolic_bp = request.form.get('systolic_bp')
            diastolic_bp = request.form.get('diastolic_bp')
            spo2 = request.form.get('spo2')
            temperature = request.form.get('temperature')
            #notes = request.form.get('notes')
            
            new_data = np.array([[69,heart_rate, systolic_bp, diastolic_bp, spo2, temperature]])
            new_data_scaled = loaded_scaler.transform(new_data)
            # Predict
            prediction = loaded_model.predict(new_data_scaled)
            print("\n🔍 Prediction for new data:", "Respiratory Disorder" if prediction[0] == 1 else "Normal")
            notes="Respiratory Disorder" if prediction[0] == 1 else "Normal"
            health_param = HealthParameter(
                user_id=current_user.id,
                heart_rate=heart_rate if heart_rate else None,
                systolic_bp=systolic_bp if systolic_bp else None,
                diastolic_bp=diastolic_bp if diastolic_bp else None,
                spo2=spo2 if spo2 else None,
                temperature=temperature if temperature else None,
                notes=notes
            )
            
            db.session.add(health_param)
            db.session.commit()
            flash('Health parameters recorded successfully!', 'success')
            return redirect(url_for('health_parameters'))
        
        # Get all health parameters for this user
        params = HealthParameter.query.filter_by(user_id=current_user.id)\
            .order_by(HealthParameter.recorded_at.desc()).all()
        
        return render_template('health_params.html', params=params)
    
    @app.route('/upload-report', methods=['GET', 'POST'])
    @login_required
    def upload_report():
        if request.method == 'POST':
            
            
            if 'report_file' not in request.files:
                flash('No file selected!', 'danger')
                return redirect(request.url)
            
            file = request.files['report_file']
            if file.filename == '':
                flash('No file selected!', 'danger')
                return redirect(request.url)
            
            if file and allowed_file(file.filename, app.config['ALLOWED_REPORT_EXTENSIONS']):
                filename = secure_filename(file.filename)
                random_hex = os.urandom(8).hex()
                _, f_ext = os.path.splitext(filename)
                filename = random_hex + f_ext
                file_path = os.path.join(app.root_path, 'static/uploads/reports', filename)
                file.save(file_path)
                result = predict_img(file_path)
                gradcam(file_path)
                report = MedicalReport(
                    user_id=current_user.id,
                    title=result,
                    description="",
                    file_path=filename
                )
                
                db.session.add(report)
                db.session.commit()
                flash('Report uploaded successfully!', 'success')
                return redirect(url_for('results'))
            else:
                flash('Invalid file type! Please upload PDF, JPG, PNG, DOC, or DOCX files.', 'danger')
        
        return render_template('upload_report.html')
    
    @app.route('/results')
    @login_required
    def results():
        reports = MedicalReport.query.filter_by(user_id=current_user.id)\
            .order_by(MedicalReport.uploaded_at.desc()).all()
        return render_template('results.html', reports=reports)
    
    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('You have been logged out.', 'info')
        return redirect(url_for('login'))
    
    # API endpoint to get health parameters data for charts
    @app.route('/api/health-data')
    @login_required
    def api_health_data():
        params = HealthParameter.query.filter_by(user_id=current_user.id)\
            .order_by(HealthParameter.recorded_at).limit(20).all()
        
        data = {
            'dates': [p.recorded_at.strftime('%Y-%m-%d %H:%M') for p in params],
            'heart_rates': [p.heart_rate for p in params if p.heart_rate],
            'systolic_bp': [p.systolic_bp for p in params if p.systolic_bp],
            'diastolic_bp': [p.diastolic_bp for p in params if p.diastolic_bp],
            'spo2': [p.spo2 for p in params if p.spo2],
            'temperature': [p.temperature for p in params if p.temperature]
        }
        
        return jsonify(data)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
