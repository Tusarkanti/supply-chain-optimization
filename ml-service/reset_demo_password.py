from app import app, db
from database_schema import User
from werkzeug.security import generate_password_hash

with app.app_context():
    user = User.query.filter_by(email='demo@supplychain.com').first()
    if user:
        user.password_hash = generate_password_hash('demo123')
        db.session.commit()
        print("Demo user password reset successfully.")
    else:
        print("Demo user not found.")
