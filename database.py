from app import create_app
from models import db, User

def init_db():
    app = create_app()
    with app.app_context():
        # Drop all tables and recreate them
        db.drop_all()
        db.create_all()
        
        # Create a sample user for testing
        if not User.query.filter_by(email='admin@example.com').first():
            admin = User(
                name='Admin User',
                email='admin@example.com',
                phone='1234567890',
                gender='Male',
                age=30,
                blood_group='O+',
                address='Sample Address',
                profile_pic='default.jpg'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Database initialized successfully!")
            print("Sample user created - Email: admin@example.com, Password: admin123")

if __name__ == '__main__':
    init_db()
