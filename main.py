from werkzeug.utils import redirect, secure_filename

from training_model.custom_ann import CustomAnn
from training_model.custom_cnn import CustomCnn
from training_model.transfer_learning import TransferLearning
import os

from flask import Flask, render_template, request, flash, url_for

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'upload_file')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#     obj = CustomAnn()
#     obj_cnn = CustomCnn()
#     obj_transfer = TransferLearning(open(os.path.join(os.getcwd(), 'logs/train.txt'), 'w+'))
#     #obj_transfer.evaluate()
#     obj_transfer.predict(os.path.join(os.getcwd(), 'data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg'))

@app.route("/")
def index():
    return render_template('index.html')


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
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('index.html')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(use_reloader=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
