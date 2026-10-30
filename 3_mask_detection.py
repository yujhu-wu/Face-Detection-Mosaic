import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('keras_model.h5', compile=False)  # 載入模型
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)          # 設定資料陣列

def text(text):      # 建立顯示文字的函式
    global img
    org = (0,50)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255,255,255)
    thickness = 2
    lineType = cv2.LINE_AA
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break

    img = cv2.resize(frame, (540, 300))  # 調整frame大小為540x300

    # 計算裁剪框的起始座標和大小
    crop_x = int((img.shape[1] - 320) / 2)
    crop_y = int((img.shape[0] - 240) / 2)
    cropped_img = img[crop_y:crop_y+240, crop_x:crop_x+320]

    resized_img = cv2.resize(cropped_img, (224, 224))

    image_array = resized_img.astype(np.float32) / 127.0 - 1
    data[0] = image_array
    prediction = model.predict(data)
    a,b,bg= prediction[0]
    print(a,b,bg)
    if a>0.700:              # 判斷沒戴口罩
        text('no mask!!')
    if b>0.300:              # 判斷有戴口罩
        text('yes mask~')
    cv2.imshow('Mask Detect', img)
    if cv2.waitKey(1) == ord('q'):
        break  # 按下'q'鍵停止

cap.release()
cv2.destroyAllWindows()
