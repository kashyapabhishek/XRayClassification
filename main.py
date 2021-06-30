from werkzeug.utils import redirect, secure_filename
from training_model.custom_ann import CustomAnn
from training_model.custom_cnn import CustomCnn
from training_model.transfer_learning import TransferLearning
from flask_session import Session
from flask import Flask, render_template, request, flash, url_for, session
import tensorflow as tf
import os

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static/upload_file')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = []


@app.route("/")
def index():
    return render_template('index.html', models=model)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            model = []
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            model.append(os.path.join('static/upload_file', filename))
            session["model"] = model
        return render_template('index.html', models=model)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        model = session["model"]
        obj_transfer = TransferLearning(open(os.path.join(os.getcwd(), 'logs/predict.txt'), 'w+'))
        prediction = obj_transfer.predict(os.path.join(os.getcwd(), model[0]))
        prediction = tf.print(tf.round(prediction))
        model.append(prediction)
    return render_template('index.html', models=model)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)

