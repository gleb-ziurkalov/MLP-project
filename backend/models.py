from flask_sqlalchemy import SQLAlchemy

database = SQLAlchemy()

class Users(database.Model):
    userid = database.Column(database.Integer, primary_key=True, autoincrement = True)
    email = database.Column(database.String(255), unique=True, nullable=False)
    username = database.Column(database.String(255), unique=True, nullable=False)
    password = database.Column(database.String(255), nullable=False)


class EvalData(database.Model):
    UploadID = database.Column(database.Integer, primary_key=True, autoincrement=True)  # Primary key
    UploadDate = database.Column(database.DateTime, nullable=False)  # Mandatory column
    InputFileName = database.Column(database.String(255), nullable=False)  # Mandatory column
    OutputFileName = database.Column(database.String(255), nullable=False)  # Mandatory column
    UserID = database.Column(database.Integer, database.ForeignKey('Users.UserID'), nullable=True) 