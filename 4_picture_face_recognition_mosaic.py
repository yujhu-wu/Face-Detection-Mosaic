import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()         # 啟用訓練人臉模型方法
recognizer.read('face.yml')                               # 讀取人臉模型檔

img = cv2.imread('25.jpg')                                # 指定要用資料夾中的其中一張圖片，若要換圖片下面第21行那裡也要改成一樣的
img = cv2.resize(img, (450,600))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # 載入人臉偵測模型
faces = face_cascade.detectMultiScale(gray,1.2,3)                 # 開始辨識影像中的人臉

mosaic_applied = True                                             # 標記是否已應用馬賽克

name = {
    '1':'[25]YU-JHU',
    '2':'[27]YUN-JHEN',
    '3':'[41]YI-JHEN',
    '4':'[46]TZU-LING'
    }

while True:
    img = cv2.imread('25.jpg')                                    # 指定要用資料夾中的其中一張圖片，若要換圖片上面第5行那裡也要改成一樣的
    img = cv2.resize(img, (450, 600))                             # 自動調整圖片大小
    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    #依序判斷每張臉屬於哪個 id
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)            # 標記人臉外框
        idnum,confidence = recognizer.predict(gray[y:y+h,x:x+w])  # 取出idnum以及信心指數
        if confidence < 65:
            text = name[str(idnum)]                               # 如果信心指數小於 65，取得對應的名字
        else:
            text = '???'                                          # 不然名字就是'???'

        cv2.putText(img, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)


        if mosaic_applied:
            mosaic = img[y:y + h, x:x + w]
            level = 15
            mh = int(h / level)
            mw = int(w / level)
            mosaic = cv2.resize(mosaic, (mw, mh), interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(mosaic, (w, h), interpolation=cv2.INTER_NEAREST)
            img[y:y + h, x:x + w] = mosaic

    cv2.imshow('Picture Mosaic', img)
    if cv2.waitKey(1) == ord('q'):
        break    # 按下'q'鍵停止，這裡要長按q或連續一直狂按直到關閉
    elif text == '[25]YU-JHU' and cv2.waitKey(0) == ord('1'):     # 按下數字鍵'1'
        mosaic_applied = not mosaic_applied                         # 切換[25號-雨茿]馬賽克應用狀態
    elif text == '[27]YUN-JHEN' and cv2.waitKey(0) == ord('2'):   # 按下數字鍵'2'
        mosaic_applied = not mosaic_applied                         # 切換27號-芸禎]馬賽克應用狀態
    elif text == '[41]YI-JHEN' and cv2.waitKey(0) == ord('3'):    # 按下數字鍵'3'
        mosaic_applied = not mosaic_applied                         # 切換[41號-奕臻]馬賽克應用狀態
    elif text == '[46]TZU-LING' and cv2.waitKey(0) == ord('4'):   # 按下數字鍵'4'
        mosaic_applied = not mosaic_applied                         # 切換[46號-子玲]馬賽克應用狀態

cv2.destroyAllWindows()
