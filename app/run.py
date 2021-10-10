import os
import argparse

from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask_dropzone import Dropzone
from PIL import Image

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Run web app')
# parser.add_argument('--database_filepath', type=str, help='Path to database')
# parser.add_argument('--model_filepath', type=str, help='Path to saved model')
args = parser.parse_args()    


app = Flask(__name__)
dropzone = Dropzone(app)

app.config.update(
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=1,
    DROPZONE_UPLOAD_ON_CLICK=True
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')

@app.route('/test')
def test():
    return render_template('test.html')

# web page that handles user query and displays model results
@app.route('/go', methods = ['POST', 'GET'])
def go():
    if request.method == 'POST':
        for key, f in request.files.items():
            if key.startswith('file'):
                # f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
                print(f)
                # image = Image.open(f)
                # plt.imshow(image)
                # plt.show()
    result = "this is output"

    return render_template(
        'go.html', result=result
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()