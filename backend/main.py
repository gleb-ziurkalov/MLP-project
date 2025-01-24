from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


from configuration import Configuration

from models import database
from models import Users
from models import EvalData

from evaluator.evaluator import pdf_to_image, classify_lines, image_to_text, json_print, read_json_as_array


from datetime import datetime

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os



app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = Configuration.SQLALCHEMY_DATABASE_URI

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


EXTRACTED_TEXT_FOLDER = './jsons/'
app.config['JSON_FOLDER'] = EXTRACTED_TEXT_FOLDER

OUTPUT_FOLDER = './output/'
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

MODEL_DIR = './evaluator/data/models/block_classification_oversampled/'


MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)



current_file = None
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

extracted_text = None
os.makedirs(EXTRACTED_TEXT_FOLDER, exist_ok=True)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
        input_file_name_arr= eval.InputFileName.split('_')
        prefix = '_'.join(input_file_name_arr[:2])
        app.logger.info(prefix)
        sufix = '_'.join(input_file_name_arr[2:]) 
        eval_history.append(
            {
                "uploadDate":eval.UploadDate,
                "documentName":sufix,
                "evaluation":eval.OutputFileName,
                "prefix":prefix
            }
        )
    return jsonify(eval_history), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_file
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files.get('file')   


    # Check if the file has a valid name
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    uid = int(request.form.get('userID'))
    count = 0
    if uid == 0:
        count = EvalData.query.filter(EvalData.UserID.is_(None)).count()
    else:
        count = EvalData.query.filter(EvalData.UserID == uid).count()

    file_save_code = "U"+str(uid)+"_E"+str(count+1)+"_"

    # Save the file to the upload folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_save_code + file.filename)
    file.save(file_path)
    current_file = file_path

    #posalji filepath nazad na front, pa ga vrati na back, i tako razmenjuj
    return jsonify({'message': f'File {file.filename} uploaded successfully'}), 200

@app.route('/extract', methods=['POST'])
def extract_file():
    global current_file, extracted_text
    pages = pdf_to_image(current_file)
    app.logger.info(pages)
    text_lines = image_to_text(pages, MODEL)
    app.logger.info(text_lines)


    extracted_text = json_print(EXTRACTED_TEXT_FOLDER,current_file.split('/')[2], text_lines, mod='text')
   
    return jsonify(text_lines)



@app.route('/evaluate', methods=['POST'])
def evaluate():
    global current_file, extracted_text
    app.logger.info(extracted_text)
    txt = read_json_as_array(extracted_text)

    classified_lines = classify_lines(MODEL, TOKENIZER, txt)
    compliance_processed = [line for line, label in classified_lines if label == 1]


    output_name = json_print(OUTPUT_FOLDER, current_file.split('/')[2], compliance_processed, mod="eval") 

    try:
        data = EvalData()
        uid_data = request.json 
        app.logger.info(uid_data.get('userID'))
        data.UserID = None if int(uid_data.get('userID')) == -1 else uid_data.get('userID')
        data.InputFileName = current_file.split('/')[2]
        data.OutputFileName= output_name.split('/')[2]
        data.UploadDate = datetime.now()
        #zavrsi ovo
        database.session.add(data)
        database.session.commit()
    except Exception as e:
        app.logger.error(repr(e))
        return jsonify({"message": "Error, coudn't save evaluation data"}), 409


    current_file = None
    extracted_text = None
    return jsonify(compliance_processed)

@app.route('/backend/uploads/<path:filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/backend/output/<path:filename>')
def serve_json(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if ( __name__ == "__main__" ):
    app.run ( host = "0.0.0.0", debug = True )