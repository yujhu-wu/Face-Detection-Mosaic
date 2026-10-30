import cv2
import numpy as np
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # 載入人臉追蹤模型
recog = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型方法
faces = []
ids = []

#face01 = 雨茿
for i in range(1,78):
    img = cv2.imread(f'face01/{i}.jpg')           # 依序開啟每一張[雨茿-沒口罩]的照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
    img_np = np.array(gray,'uint8')               # 轉換成指定編碼的numpy陣列
    face = detector.detectMultiScale(gray)        # 擷取人臉區域
    for(x,y,w,h) in face:
        faces.append(img_np[y:y+h,x:x+w])         # 記錄人臉的位置和大小內像素的數值
        ids.append(1)                             # 記錄[雨茿-沒口罩]人臉對應的id，1表示[雨茿-沒口罩]的id為1

#face02 = 芸禎
for i in range(1,155):
    img = cv2.imread(f'face02/{i}.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_np = np.array(gray,'uint8')
    face = detector.detectMultiScale(gray)
    for(x,y,w,h) in face:
        faces.append(img_np[y:y+h,x:x+w])
        ids.append(2)                             # 記錄[芸禎-沒口罩]人臉對應的id，2表示[芸禎-沒口罩]的id為 2

#face03 = 奕臻
for i in range(1,150):
    img = cv2.imread(f'face03/{i}.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_np = np.array(gray,'uint8')
    face = detector.detectMultiScale(gray)
    for(x,y,w,h) in face:
        faces.append(img_np[y:y+h,x:x+w])
        ids.append(3)                             # 記錄[奕臻-沒口罩]人臉對應的id，3表示[奕臻-沒口罩]的id為3

#face04 = 子玲
for i in range(1,44):
    img = cv2.imread(f'face04/{i}.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_np = np.array(gray,'uint8')
    face = detector.detectMultiScale(gray)
    for(x,y,w,h) in face:
        faces.append(img_np[y:y+h,x:x+w])
        ids.append(4)                             # 記錄[子玲-沒口罩]人臉對應的id，4表示[子玲-沒口罩]的id為4


print('training...')                              # 提示開始訓練
recog.train(faces,np.array(ids))                  # 開始訓練
recog.save('face.yml')                            # 訓練完成儲存為'face.yml'
print('ok!')                                      # 最後顯示'ok!'
