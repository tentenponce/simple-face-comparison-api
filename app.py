#!/usr/bin/env python

import os

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
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def get_ext(filename):
    return os.path.splitext(filename)[1]


def get_percentage(result):
    return 100 - (result * 100)


# @param unknown_face image of the face in the ID
# @param  known_faces to known_face9 image of the face to compare
@app.route('/compare', methods=['POST'])
def compare_faces():
    # create tmp folder
    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    # save unknown face
    unknown_face = request.files['unknown_face']
    unknown_face_filename = 'unknown_face' + get_ext(unknown_face.filename)
    unknown_face.save(os.path.join(app.config['UPLOAD_FOLDER'], unknown_face_filename))
    unknown_face_file = face_recognition.load_image_file(
        os.path.join(app.config['UPLOAD_FOLDER'], unknown_face_filename))

    try:
        unknown_face_encoded = face_recognition.face_encodings(unknown_face_file)[0]
    except IndexError:
        response = jsonify({'message': 'Please check the quality of your ID.'})
        response.status_code = 400
        return response

    # save known faces
    known_faces = request.files.getlist('known_faces')
    known_faces_encoded = []
    i = 0
    for known_face in known_faces:
        if valid_known_face(known_face.filename):
            known_face_filename = 'known_face' + str(i) + get_ext(known_face.filename)
            known_face.save(os.path.join(app.config['UPLOAD_FOLDER'], known_face_filename))
            known_face_file = face_recognition.load_image_file(
                os.path.join(app.config['UPLOAD_FOLDER'], known_face_filename))

            try:
                known_faces_encoded.append(face_recognition.face_encodings(known_face_file)[0])
            except IndexError:
                pass

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

    return jsonify(list(map(get_percentage, results.tolist())))


if __name__ == '__main__':
    app.run(debug=True)
