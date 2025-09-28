import sys
sys.path.append('ml-service')
from app import app, db
from database_schema import User

with app.app_context():
    users = User.query.all()
    for user in users:
        print(f"Email: {user.email}")
        print(f"Password hash: '{user.password_hash}'")
        if user.password_hash is None:
            print("HASH IS NONE!")
        elif user.password_hash == '':
            print("HASH IS EMPTY STRING!")
        else:
            try:
                from werkzeug.security import check_password_hash
                result = check_password_hash(user.password_hash, 'test')
                print(f"Check result: {result}")
            except Exception as e:
                print(f"Error checking hash: {e}")
        print("---")
