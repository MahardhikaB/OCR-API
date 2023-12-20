import cv2
import json
import numpy as np
import ocr
import time
# Note: Uncomment for YOLO feature
# import yolo_detect
from flask import Flask, request

app = Flask(__name__)

@app.route('/ocr', methods = ['POST'])
def upload_file():
    start_time = time.time()

    if 'image' not in request.files:
        finish_time = time.time() - start_time

        json_content = {
            'message': "image is empty",
            'time_elapsed': str(round(finish_time, 3))
        }
    else:
        imagefile = request.files['image'].read()
        npimg = np.frombuffer(imagefile, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        cv2.imwrite('image.jpg', image)
        # Note: Uncomment for YOLO feature
        # image = yolo_detect.main(image)
        nik, nama, tempat_lahir, tgl_lahir, jenis_kelamin, gol_darah, alamat, agama, status_perkawinan, pekerjaan, kewarganegaraan  = ocr.main(image)

        finish_time = time.time() - start_time

        json_content = {
            'nik': str(nik),
            'nama': str(nama),
            'jenis_kelamin': str(jenis_kelamin),
            'alamat': str(alamat),
            'time_elapsed': str(round(finish_time, 3))
        }

    python2json = json.dumps(json_content)
    return app.response_class(python2json, content_type = 'application/json')

if __name__ == "__main__":
    app.run(host = '0.0.0.0')
