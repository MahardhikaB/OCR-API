import cv2
import numpy as np
import os
import pandas as pd
import pytesseract
import re
import textdistance
import datetime
from datetime import date
from operator import itemgetter, attrgetter

def ocr_raw(image):
    image = cv2.resize(image, (750, 50 * 16))

    # crop the image to get the identity text only
    image = image[20:600, 180:580]

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    # img_gray = cv2.fastNlMeansDenoising(img_gray, None, 3, 7, 21)
    # cv2.fillPoly(img_gray, pts=[np.asarray([(540, 150), (540, 499), (798, 499), (798, 150)])], color=(255, 255, 255))
    th, threshed = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
    result_raw = pytesseract.image_to_string(threshed, lang="ind")

    return result_raw

def main():
    result_raw = ocr_raw(cv2.imread('image.jpg'))

    nik = ""
    nama = ""
    tempat_lahir = ""
    tgl_lahir = ""
    jenis_kelamin = ""
    gol_darah = ""
    alamat = ""
    agama = ""
    status_perkawinan = ""
    pekerjaan = ""
    kewarganegaraan = ""

    # remove empty lines
    lines = list(filter(lambda x: x != '', result_raw.split('\n')))
    
    # remove any ':' and '1' and '2' character in the beggining of the string
    lines = list(map(lambda x: x.strip(':12'), lines))
    
    # remove any empty space at the beginning and end of the string
    lines = list(map(lambda x: x.strip(), lines))
    
    for i in lines:
        print(i)
        
    # Find NIK and its index
    nik_index = next((i for i, line in enumerate(lines) if re.match(r'^\d+$', line)), None)
    if nik_index is not None:
        nik = lines[nik_index]
        nik_index += 1
        if nik_index < len(lines):
            nama = lines[nik_index]
            nik_index += 1
        if nik_index < len(lines):
            # check if the line contain any number, if not append it to name, if yes then it's the birth date
            if not any(char.isdigit() for char in lines[nik_index]):
                nama += ' ' + lines[nik_index]
                nik_index += 1
                if nik_index < len(lines):
                    # split lines into tempat_lahir and tgl_lahir by checking ',' character
                    if ',' in lines[nik_index]:
                        tempat_lahir, tgl_lahir = [part.strip() for part in lines[nik_index].split(',')]
                        nik_index += 1
                    else:
                        tempat_lahir = lines[nik_index]
                        nik_index += 1
                        if nik_index < len(lines):
                            tgl_lahir = lines[nik_index]
                            nik_index += 1
            else:
                tgl_lahir = lines[nik_index]
                nik_index += 1
        if nik_index < len(lines):
            jenis_kelamin = lines[nik_index]
            # trim only the first 9 letters
            jenis_kelamin = jenis_kelamin[:9]
            nik_index += 1
        if nik_index < len(lines):
            # get the address of 'alamat' from the next 3 index
            alamat = ' '.join(lines[nik_index:nik_index+4])
            nik_index += 4
        if nik_index < len(lines):
            agama = lines[nik_index]
            nik_index += 1
        if nik_index < len(lines):
            status_perkawinan = lines[nik_index]
            nik_index += 1
        if nik_index < len(lines):
            pekerjaan = lines[nik_index]
    else:
        print('not found')
        
    if "D" in nik:
        nik = nik.replace("D", "0")
    if "?" in nik:
        nik = nik.replace("?", "7")
    if "L" in nik:
        nik = nik.replace("L", "1")
    if "O" in nik:
        nik = nik.replace("O", "0")
    if "S" in nik:
        nik = nik.replace("S", "5")

    print('nik: ' + nik)
    print('nama: ' + nama)
    print('tgl_lahir: ' + tgl_lahir)
    print('tempat_lahir: ' + tempat_lahir)
    print('jenis_kelamin: ' + jenis_kelamin)
    print('alamat: ' + alamat)
    print('agama: ' + agama)
    print('status_perkawinan: ' + status_perkawinan)
    print('pekerjaan: ' + pekerjaan)

    return nik, nama, tempat_lahir, tgl_lahir, jenis_kelamin, gol_darah, alamat, agama, status_perkawinan, pekerjaan, kewarganegaraan

if __name__ == '__main__':
    main(sys.argv[1])
