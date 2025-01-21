from flask_sqlalchemy import SQLAlchemy

database = SQLAlchemy()

class Users(database.Model):
    id = database.Column(database.Integer, primary_key=True)
    email = database.Column(database.String(255), unique=True, nullable=False)
    username = database.Column(database.String(255), unique=True, nullable=False)
    password = database.Column(database.String(255), nullable=False)