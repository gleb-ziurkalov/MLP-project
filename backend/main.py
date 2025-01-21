from flask import Flask, request, jsonify
from flask_cors import CORS


from configuration import Configuration

from models import database
from models import Users



app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = Configuration.SQLALCHEMY_DATABASE_URI

database.init_app(app)

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    user = Users.query.filter(Users.email == email, Users.password == password).first()

    if user:
        return jsonify({
            'email':user.email,
            'password':user.password,
            'username':user.username
        }), 200
    else:
        return jsonify({"message": "Invalid credentials"}), 401

@app.route('/signup', methods=['POST'])
def signup():

    try:
        data = request.json
        user = Users()
        user.email = data['email']
        user.username= data['username']
        user.password = data['password']
        database.session.add(user)
        database.session.commit()
        return jsonify({"message":"Signup complete!"}), 201
    except Exception as e:
        return jsonify({"message": "Error, email or username already exist"}), 409


if ( __name__ == "__main__" ):
    app.run ( host = "0.0.0.0", debug = True )