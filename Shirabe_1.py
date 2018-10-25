#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 13:25:27 2018

@author: abetakuto
"""
"""
現状，iphoneで撮影した画像を利用しています．
webカメラ等別のカメラで撮影した場合は細かいパラメータ調整が必要となります．
調べたい単語の左下に指先をおいて撮影した画像を入力すれば，その周辺の単語を翻訳した結果を出力します．

参考url
https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
https://postd.cc/image-processing-101/

"""




#-------------------　ライブラリのインポート -------------------------------
import requests
from io import BytesIO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import os
import matplotlib
from PIL import Image
from googletrans import Translator
import subprocess
from time import sleep
"""
google transはそのまま使うとエラーが生じるので以下urlを参考にパッチを利用する
https://stackoverflow.com/questions/52455774/googletrans-stopped-working-with-error-nonetype-object-has-no-attribute-group
"""



#肌色の領域のみを取り出しマスク画像を生成する関数
def mask(piet):
    piet_hsv = cv2.cvtColor(piet, cv2.COLOR_BGR2HSV)

    skin_min = np.array([0, 60, 80], np.uint8)
    skin_max = np.array([30, 255, 255], np.uint8)

    threshold_skin_img = cv2.inRange(piet_hsv, skin_min, skin_max)

    threshold_skin_img = cv2.cvtColor(threshold_skin_img, cv2.COLOR_GRAY2RGB)

    return(threshold_skin_img)


# 画像をグレースケールに変換し、そこにガウスぼかしを適用して単純化とノイズの除去を行う
# これは前処理の一般的な形式で、画像を扱う際は、大抵の場合、これを最初に行うらしい
finger = cv2.imread('sample.jpg')
finger_mask = mask(finger)
finger_gray = cv2.cvtColor(finger_mask, cv2.COLOR_BGR2GRAY)
finger_preprocessed = cv2.GaussianBlur(finger_gray, (5, 5), 0)

# さらに細かいノイズ(点)を除去
kernel = np.ones((60,60),np.uint8)
finger_clean = cv2.morphologyEx(finger_preprocessed, cv2.MORPH_OPEN, kernel)

#前処理を行った画像に2値化閾値処理を適用
thresh = cv2.threshold(finger_clean, 45, 255, cv2.THRESH_BINARY)[1]


# 指の輪郭を検出，finger_contoursに輪郭の座標が格納される
contour_img, finger_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

try :
    cnts = finger_contours[0] if imutils.is_cv2() else finger_contours[1]
except:
    cnts = finger_contours[0] #if imutils.is_cv2() else finger_contours[1]
#cnts = finger_contours[0] #if imutils.is_cv2() else finger_contours[1]
c = max(cnts, key=cv2.contourArea)


#y座標の昇順で並び替え
sorted_cnts = cnts[:][:,0][np.argsort(cnts[:][:,0][:,1])]

#輪郭のtop100点を格納
top100_lst = sorted_cnts[:100,:]

#x座標の昇順に並び替え
sorted_top100_lst = top100_lst[np.argsort(top100_lst[:,0])]


#指先の左端と右端の座標を保存
left_side = tuple(sorted_top100_lst[0])
right_side = tuple(sorted_top100_lst[-1])


# finger_and_contoursに指の左端と右端に印をつけた画像をコピー
# 可視化する際は　plt.imshow(finger_and_contours)
finger_and_contours = np.copy(finger)


#cv2.drawContours(finger_and_contours, large_contours, -1, (255,0,0))
cv2.circle(finger_and_contours, left_side, 50, (255, 0, 0), -1)
cv2.circle(finger_and_contours, right_side, 50, (255, 0, 0), -1)


# ----------   画像をトリミング --------------
# height，widthは今後調整する必要あり
height = right_side[0]-left_side[0]
width = (right_side[0]-left_side[0])*3

y = left_side[1]
x = left_side[0]-50

dstImg = finger[y-height:y,x:x+width]

cv2.imwrite('trimming.png',dstImg)


# --------------------------------  Azureの利用 -------------------------------
#--------------------- API情報を入力 -----------------------------------
# キーを入力
key_file = open('../key/key_cog.txt')
key = key_file.read()
key_file.close()

subscription_key = key
assert subscription_key

vision_base_url = "https://eastasia.api.cognitive.microsoft.com/vision/v1.0/"
analyze_url = vision_base_url + "ocr"



#----------------------- 画像データの指定 -----------------------------------
"""
#----1. webから取得する場合 ----#
#画像のあるURLを指定
image_file = "https://s8.favim.com/610/150406/butterfly-couple-cute-cute-text-Favim.com-2626029.jpg"
image_data  = {'url': image_file}

headers = {'Ocp-Apim-Subscription-Key': subscription_key }
"""

#"""
#----2. Localから取得する場合 ----#
# 画像へのパスをimage_fileに代入
image_file = "trimming.png"
image_data = open(image_file, "rb").read()

headers    = {'Ocp-Apim-Subscription-Key': subscription_key,
              'Content-Type': 'application/octet-stream'}
#"""
#------------------ 画像から取得したい情報の指定 -----------------------------------

params   = {'language': 'unk', 'detectOrientation ': 'true'}

#------------------ httpにリクエストを送る(データを取ってくる) -----------------------
"""
#----1. web画像の場合 ----#
response = requests.post(analyze_url, headers=headers, params=params, json=image_data)

"""
#"""
#----2. Local画像の場合 ----#
response = requests.post(
    analyze_url, headers=headers, params=params, data=image_data)
#"""

response.raise_for_status()

#analysisに画像解析結果のjsonファイルを格納
analysis = response.json()


#------------------- 結果の表示 -----------------------------------
#print(analysis)


#######################################################################
###########   ここから翻訳　　###########
#######################################################################

#画像中に含まれる単語を全てlstに保存する．
lst = []
region = []
for line in analysis["regions"][0]['lines']:
    for word in line["words"]:
        lst.append(word["text"])
        region.append(word["boundingBox"])

#単語のx座標を保存する．
x_region = []
for i in region:
    x_region.append(int(i.split(',')[0]))

target_x = 10000
target_idx = 0
for n, i in enumerate(x_region):
    if i >= 40:
        num = min(abs(target_x - 50),abs(i - 50))
        if num == abs(i-50):
            target_x = i
            target_idx = n
    else:
        pass

lst = lst[target_idx]

strip_lst = [";", ",", ".", ":"]
del_str = "a the an i my me mine you your yours he his him she her hers "+\
"they their them theirs it its we our us ours to"
del_lst = del_str.split(" ")

# 翻訳する単語を一文字に絞れなかった場合は以下の処理を実行
if type(lst) == 'list':
    lst = list(map(lambda x : x.lower(), lst))


    #冠詞，人称代名詞，不定詞および，一文字の場合はリストから削除する．
    lst = list(filter(lambda x: len(x) != 1, lst))
    lst = list(filter(lambda x: x not in del_lst, lst))
    for i in strip_lst:
        lst = list(map(lambda x: x.replace(i, ""), lst))

# lstが1単語であった場合は記号のみを翻訳する文字から削除する
for i in strip_lst:
    lst = lst.replace(i, "")

#Azureに投げた画像を表示
plt.imshow(dstImg)

#実際に翻訳
translator = Translator()


#翻訳した順番をkeyにして単語と意味を格納
result_dict = {}
if type(lst) == 'list':
    for num, i in enumerate(lst):
        meaning = translator.translate(text=i, dest='ja').text
        result_dict[num] = {"en" : i, "ja" : meaning}
        print(i)
        print(meaning)
        print('\n')

else:
    meaning = translator.translate(text=lst, dest='ja').text
    result_dict[0] = {"en" : lst, "ja" : meaning}
    print(lst)
    print(meaning)
    print('\n')


# -------------- 調べた単語の発音をwavファイルで"en_sound"ディレクトリに保存する．
#en_soundが存在するば削除する.
os.system('rm -rf en_sound')
os.system('mkdir en_sound')
if type(lst) == 'list':
    for num, i in enumerate(lst):
        file_name = 'en_sound/en_{}.wav'.format(num)
        os.system('espeak ' + i + ' -w ' + file_name)
else:
    file_name = 'en_sound/en_0.wav'
    os.system('espeak ' + lst + ' -w ' + file_name)



"""
# 文章を入力する場合の参考
text = "Hello world."
text_lst = text.split(" ")
speak_text = "\ ".join(text_lst)
speak_text = " " + speak_text
"""


# -------------- 調べた単語の意味をwavファイルで"ja_sound"ディレクトリに保存する．
def jtalk(t, num):
    open_jtalk = ['open_jtalk']
    mech = ['-x', '/usr/local/Cellar/open-jtalk/1.10_1/dic']
    htsvoice = ['-m', '/usr/local/Cellar/open-jtalk/1.10_1/voice/mei/mei_normal.htsvoice']
    speed = ['-r', '0.8']
    outwav = ['-ow', 'ja_sound/ja_{}.wav'.format(num)]
    cmd = open_jtalk + mech + htsvoice + speed + outwav
    c = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    c.stdin.write(t)
    c.stdin.close()
    c.wait()
    # 音声を再生する場合
    aplay = ['afplay', 'ja_sound/ja_{}.wav'.format(num)]
    wr = subprocess.Popen(aplay)

os.system('rm -rf ja_sound')
os.system('mkdir ja_sound')
for num in result_dict.keys():
    jtalk(result_dict[num]['ja'].encode('utf-8'), num)
    sleep(1)
