import sys
sys.path.append('ml-service')
from app import app, db, User

with app.app_context():
    user = User.query.filter_by(email='admin@supplychain.com').first()
    print('User found:', user is not None)
    if user:
        print('Password hash:', user.password_hash[:20] + '...')
        print('Check password:', user.check_password('admin'))
