from passlib.hash import bcrypt
hash_value = "$2b$12$.HntdWN6I1p/TTjncmRkD.fdv97mziWYmn55IfHfOEwKaElQ4AaPO"
print(bcrypt.verify('admin', hash_value))
