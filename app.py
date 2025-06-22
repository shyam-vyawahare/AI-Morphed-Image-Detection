from flask import Flask, request, jsonify, render_template, url_for, session, redirect, flash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = '9156620'

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    full_name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MODEL_PATH = 'saved_model/morph_detection_model.keras'
THRESHOLD = 0.7

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
try:
    def loss_fn(y_true, y_pred): return y_pred  # Dummy function
    model = load_model(MODEL_PATH, custom_objects={'loss_fn': loss_fn}, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        raw_p = model.predict(img, verbose=0)[0][0]

        if raw_p < THRESHOLD:
            status = "MORPHED"
            confidence = 100 * (THRESHOLD - raw_p) / THRESHOLD
        else:
            status = "ORIGINAL"
            confidence = 100 * (raw_p - THRESHOLD) / (1.0 - THRESHOLD)

        confidence = float(round(min(max(confidence, 0.0), 100.0), 2))

        return {
            "status": status,
            "raw_output": float(raw_p)
        }

    except Exception as e:
        return {"error": str(e)}

@app.route('/', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file selected"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            result = predict_image(save_path)
            if 'error' in result:
                return jsonify({"error": result['error']}), 500

            stamp_filename = 'approved.png' if result['status'] == 'ORIGINAL' else 'disapproved.png'

            return jsonify({
                "status": result['status'],
                "image_url": url_for('static', filename=f'uploads/{filename}'),
                "stamp_url": url_for('static', filename=stamp_filename)
            })

        return jsonify({"error": "Invalid file type"}), 400

    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('upload_file'))
        
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user is None or not user.check_password(password):
            error = 'Invalid username or password'
        else:
            login_user(user)
            return redirect(url_for('upload_file'))
    
    return render_template('login.html', error=error)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('upload_file'))
        
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        full_name = request.form.get('full_name')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not all([username, email, full_name, password, confirm_password]):
            error = 'All fields are required'
        elif password != confirm_password:
            error = 'Passwords do not match'
        elif User.query.filter_by(username=username).first() is not None:
            error = 'Username already taken'
        elif User.query.filter_by(email=email).first() is not None:
            error = 'Email already registered'
        else:
            try:
                # Create new user
                new_user = User(
                    username=username,
                    email=email,
                    full_name=full_name,
                    is_admin=False  # Default to regular user
                )
                new_user.set_password(password)
                db.session.add(new_user)
                db.session.commit()
                
                # Automatically log in the new user
                login_user(new_user)
                return redirect(url_for('upload_file'))
            except Exception as e:
                db.session.rollback()
                error = f'Error creating account: {str(e)}'
                print(f"Error during signup: {e}")  # This will print to your console
    
    return render_template('signup.html', error=error)
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)