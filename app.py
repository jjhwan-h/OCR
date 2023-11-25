from flask import Flask, request, render_template_string
from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
import matplotlib.pyplot as plt
import pytesseract
import imutils
import cv2
import re
import requests
import numpy as np
import base64


def plt_imshow(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()

def make_scan_image(image, width, ksize=(5,5), min_threshold=20, max_threshold=60):
  image_list_title = []
  image_list = []

  org_image = image.copy()
  image = imutils.resize(image, width=width)
  ratio = org_image.shape[1] / float(image.shape[1])

  # 이미지를 grayscale로 변환하고 blur를 적용
  # 모서리를 찾기위한 이미지 연산
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, ksize, 0)
  edged = cv2.Canny(blurred, min_threshold, max_threshold)

  image_list_title = ['gray', 'blurred', 'edged']
  image_list = [gray, blurred, edged]

  # contours를 찾아 크기순으로 정렬
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

  findCnt = None

  # 정렬된 contours를 반복문으로 수행하며 4개의 꼭지점을 갖는 도형을 검출
  for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # contours가 크기순으로 정렬되어 있기때문에 제일 첫번째 사각형을 영역으로 판단하고 break
    if len(approx) == 4:
      findCnt = approx
      break


  # 만약 추출한 윤곽이 없을 경우 오류
  if findCnt is None:
    raise Exception(("Could not find outline."))


  output = image.copy()
  cv2.drawContours(output, [findCnt], -1, (0, 255, 0), 2)

  image_list_title.append("Outline")
  image_list.append(output)

  # 원본 이미지에 찾은 윤곽을 기준으로 이미지를 보정
  transform_image = four_point_transform(org_image, findCnt.reshape(4, 2) * ratio)

  plt_imshow(image_list_title, image_list)
  plt_imshow("Transform", transform_image)
  plt_imshow("edged",edged)

  return transform_image

def replace_capsule_in_list(word_list):
    # 각 원소에 대해 '캡슬'을 '캡슐'로 변환
    corrected_list = [re.sub(r'\b\w*(?:캡술|캡슬)\b', word.replace('캡슬','캡슐').replace('캠술','캠슐'), word) for word in word_list]
    return corrected_list

def replace_capsule(text):
    # 정규표현식을 사용하여 '캡술' 또는 '캡슬'을 '캡슐'로 변환
    corrected_text = re.sub(r'\b(캡술|캡슬)\b', '캡슐', text)
    return corrected_text

app = Flask(__name__)

@app.route('/', methods=['POST'])
def hello():
    #print(request.files['url'].read())
    # try:
    #     print(request)
    #     # Assuming the image is sent as a base64-encoded string in the 'img' parameter
    #     img_data = request.form.get('img')

    #     # Check if the image is in JPEG format
    #     if 'jpeg' in request.content_type.lower():
    #         # Handle JPEG images differently if needed
    #         # For now, we'll assume it's already base64 encoded
    #         img_base64 = img_data
    #     else:
    #         # Decode the base64 string to bytes for other formats (assuming it's PNG)
    #         img_bytes = base64.b64decode(img_data)

    #         # Convert bytes to base64 for displaying in HTML
    #         img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
    # except Exception as e:
    #     return f"Error: {e}"
    #url = request.form['url']
    url = request.files['url'].read()
    # image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
    image_nparray = np.asarray(bytearray(url), dtype=np.uint8)
    print(image_nparray)
    org_image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
    # org_image2 = org_image.copy() 
    receipt_image = make_scan_image(org_image, width=1000, ksize=(5, 5), min_threshold=20, max_threshold=70)
    options = "--psm 6"
    text = pytesseract.image_to_string(cv2.cvtColor(receipt_image, cv2.COLOR_BGR2RGB), config=options, lang='kor')

    # OCR결과 출력
    # print("[INFO] OCR결과:")
    # print("==================")
    # print(text)
    # print("\n")

    pattern = r'\b\w*(?:정|캡슐|캡술|캡슬|시럽)\b'
    matches = re.findall(pattern, text)
    print(matches)

    modify_matches = replace_capsule_in_list(matches)
    print(modify_matches)  
    return modify_matches

if __name__ == '__main__':
    app.run()