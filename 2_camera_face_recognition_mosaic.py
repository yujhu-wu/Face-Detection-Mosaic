import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()         # 啟用訓練人臉模型方法
recognizer.read('face.yml')                               # 讀取人臉模型檔
cascade_path = "haarcascade_frontalface_default.xml"      # 載入人臉追蹤模型
face_cascade = cv2.CascadeClassifier(cascade_path)        # 啟用人臉追蹤

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

mosaic_applied = True  # 標記是否已應用馬賽克

while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    img = cv2.resize(img,(810,450))              # 固定尺寸
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # 轉換成黑白
    faces = face_cascade.detectMultiScale(gray)  # 追蹤人臉

    name = {
        '1':'[25]YU-JHU',
        '2':'[27]YUN-JHEN',
        '3':'[41]YI-JHEN',
        '4':'[46]TZU-LING'
    }

    # 依序判斷每張臉屬於哪個 id
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)            # 標記人臉外框
        idnum,confidence = recognizer.predict(gray[y:y+h,x:x+w])  # 取出idnum以及信心指數
        if confidence < 65:
            text = name[str(idnum)]                               # 如果信心指數小於65，取得對應的名字
        else:
            text = '???'                                          # 不然名字就是'???'

        cv2.putText(img, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        key = cv2.waitKey(1)
        if key == ord('1'):                                       # 按下數字鍵'1'
            mosaic_applied = not mosaic_applied                   # 切換馬賽克應用狀態

        if mosaic_applied:
            if text == '[25]YU-JHU':                              # 指定人物[25號-雨茿]自動打上馬賽克，其他人都不打
                mosaic = img[y:y + h, x:x + w]
                level = 15
                mh = int(h / level)
                mw = int(w / level)
                mosaic = cv2.resize(mosaic, (mw, mh), interpolation=cv2.INTER_LINEAR)
                mosaic = cv2.resize(mosaic, (w, h), interpolation=cv2.INTER_NEAREST)
                img[y:y + h, x:x + w] = mosaic

    cv2.imshow('Camera Mosaic', img)
    if cv2.waitKey(1) == ord('q'):
        break    # 按下'q'鍵停止
cap.release()
cv2.destroyAllWindows()
