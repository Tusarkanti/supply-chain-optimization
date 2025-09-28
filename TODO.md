# Login Issue Correction Plan

## Tasks
- [x] Remove account locking check from login endpoint in ml-service/app.py
- [x] Remove failed attempt increment logic from login endpoint
- [x] Fix password hashing consistency (use passlib for both set and check)
- [x] Reset admin password with consistent hashing
- [x] Test login functionality after changes
- [x] Unlock any currently locked accounts using unlock_admin.py

## Information Gathered
- Login system had account locking after 5 failed attempts for 15 minutes
- User model had account_locked, failed_login_attempts, lockout_until fields
- Login endpoint checked for lock and incremented attempts on wrong password
- Password hashing was inconsistent: set_password used werkzeug, check_password used passlib
- This caused login failures even with correct credentials

## Plan
- Modify /api/login endpoint to remove is_account_locked() check
- Remove increment_failed_attempts() call on wrong password
- Update set_password() to use passlib for consistency
- Reset admin password to use consistent hashing
- Run unlock script to unlock any existing locked accounts
