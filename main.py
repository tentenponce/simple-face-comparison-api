import os
import re

import face_recognition
from flask import Flask, request, jsonify

app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'tmp/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


# For a given file, return whether it's an allowed type or not
# and check if valid name: known_face0 to known_Face9. Maximum of
# 10 images only starting from 0 to 9.
def valid_known_face(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS'] and \
           bool(re.search('known_face[1-9]$', filename))


def get_ext(filename):
    return os.path.splitext(filename)[1]


# @param unknown_face image of the face in the ID
# @param  known_faces to known_face9 image of the face to compare
@app.route('/compare', methods=['POST'])
def compare_faces():
    # save unknown face
    unknown_face = request.files['unknown_face']
    unknown_face_filename = 'unknown_face' + get_ext(unknown_face.filename)
    unknown_face.save(os.path.join(app.config['UPLOAD_FOLDER'], unknown_face_filename))
    unknown_face_file = face_recognition.load_image_file(
        os.path.join(app.config['UPLOAD_FOLDER'], unknown_face_filename))
    unknown_face_encoded = face_recognition.face_encodings(unknown_face_file)[0]

    # save known faces
    known_faces = request.files.getlist('known_faces')
    known_faces_encoded = []
    i = 0
    for known_face in known_faces:
        known_face_filename = 'known_face' + str(i) + get_ext(known_face.filename)
        known_face.save(os.path.join(app.config['UPLOAD_FOLDER'], known_face_filename))
        known_face_file = face_recognition.load_image_file(
            os.path.join(app.config['UPLOAD_FOLDER'], known_face_filename))
        known_faces_encoded.append(face_recognition.face_encodings(known_face_file)[0])
        i += 1

    results = face_recognition.face_distance(known_faces_encoded, unknown_face_encoded)

    # remove tmp files
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    return jsonify(results=results.tolist())


if __name__ == '__main__':
    app.run(debug=True)
