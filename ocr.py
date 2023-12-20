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

def convertScale(img, alpha, beta):
    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)

def automatic_brightness_and_contrast(image, clip_hist_percent=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    # auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result

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

def strip_op(result_raw):
    result_list = result_raw.split('\n')
    new_result_list = []

    for tmp_result in result_list:
        if tmp_result.strip(' '):
            new_result_list.append(tmp_result)

    return new_result_list

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes

def return_id_number(image, img_gray):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, rectKernel)

    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)

    threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = threshCnts
    cur_img = image.copy()
    cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
    copy = image.copy()

    locs = []
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)

        # ar = w / float(h)
        # if ar > 3:
        # if (w > 40 ) and (h > 10 and h < 20):
        if h > 10 and w > 100 and x < 300:
            img = cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            locs.append((x, y, w, h, w * h))

    locs = sorted(locs, key=itemgetter(1), reverse=False)

    # nik = image[locs[1][1] - 15:locs[1][1] + locs[1][3] + 15, locs[1][0] - 15:locs[1][0] + locs[1][2] + 15]
    # text = image[locs[2][1] - 10:locs[2][1] + locs[2][3] + 10, locs[2][0] - 10:locs[2][0] + locs[2][2] + 10]

    check_nik = False

    try:
        nik = image[locs[1][1] - 15:locs[1][1] + locs[1][3] + 15, locs[1][0] - 15:locs[1][0] + locs[1][2] + 15]
        check_nik = True
    except Exception as e:
        print(e)

    if check_nik == True:
        img_mod = cv2.imread("data/module2.png")

        ref = cv2.cvtColor(img_mod, cv2.COLOR_BGR2GRAY)
        ref = cv2.threshold(ref, 66, 255, cv2.THRESH_BINARY_INV)[1]

        refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refCnts = sort_contours(refCnts, method="left-to-right")[0]

        digits = {}
        for (i, c) in enumerate(refCnts):
            (x, y, w, h) = cv2.boundingRect(c)
            roi = ref[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))
            digits[i] = roi

        gray_nik = cv2.cvtColor(nik, cv2.COLOR_BGR2GRAY)
        group = cv2.threshold(gray_nik, 127, 255, cv2.THRESH_BINARY_INV)[1]

        digitCnts, hierarchy_nik = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nik_r = nik.copy()
        cv2.drawContours(nik_r, digitCnts, -1, (0, 0, 255), 3)

        gX = locs[1][0]
        gY = locs[1][1]
        gW = locs[1][2]
        gH = locs[1][3]

        ctx = sort_contours(digitCnts, method="left-to-right")[0]

        locs_x = []
        for (i, c) in enumerate(ctx):
            (x, y, w, h) = cv2.boundingRect(c)
            if h > 10 and w > 10:
                img = cv2.rectangle(nik_r, (x, y), (x + w, y + h), (0, 255, 0), 2)
                locs_x.append((x, y, w, h))


        output = []
        groupOutput = []

        for c in locs_x:
            (x, y, w, h) = c
            roi = group[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))

            scores = []
            for (digit, digitROI) in digits.items():
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)

            groupOutput.append(str(np.argmax(scores)))

        cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
        cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        output.extend(groupOutput)
        return ''.join(output)
    else:
        return ""

def main(image):
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
