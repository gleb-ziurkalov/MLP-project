from flask import Flask, request, jsonify
from flask_cors import CORS


from configuration import Configuration

from models import database
from models import Users
from models import EvalData

import os



app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = Configuration.SQLALCHEMY_DATABASE_URI

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

current_file = None
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
            'username':user.username,
            'id':user.userid
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
        app.logger.error(repr(e))
        return jsonify({"message": "Error, email or username already exist"}), 409
    
@app.route('/history',methods=['GET'])    
def history():
    user_id = request.args.get('UserID')

    if not user_id:
        return jsonify({"error": "UserID parameter is required"}), 400
    
    eval_history = []

    query = EvalData.query.filter(EvalData.UserID == int(user_id))

    for eval in query:
        eval_history.append(
            {
                "uploadDate":eval.UploadDate,
                "documentName":eval.InputFileName,
                "evaluation":eval.OutputFileName
            }
        )
    return jsonify(eval_history), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_file
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # Check if the file has a valid name
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    # Save the file to the upload folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    current_file = file_path

    return jsonify({'message': f'File {file.filename} uploaded successfully'}), 200

@app.route('/extract', methods=['POST'])
def extract_file():
    pass

@app.route('/evaluate', methods=['POST'])
def evaluate():
    pass


if ( __name__ == "__main__" ):
    app.run ( host = "0.0.0.0", debug = True )