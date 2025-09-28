from passlib.hash import bcrypt
stored_hash = "$2b$12$riqAN7L2n2nbc5NP1yrJDuj2fi0c98IEDI4BZvIdX.zjN7TPv4BdW"
print(bcrypt.verify('admin', stored_hash))
