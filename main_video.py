

import cv2
# 加载模型
FaceCascade = cv2.CascadeClassifier('./moudel/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('./testvideo/test_video.mp4')  # 读取视频

flag=cap.isOpened()

if flag==False :
    print("video not found")
else:
    print("video found")

    while flag==True :
        ret, frame = cap.read()
        if ret == True :

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将帧转换为灰度图

            #人脸检测参数
            faces = FaceCascade.detectMultiScale(gray, 1.3, 5)  # 检测人脸

            if faces is  None:
                print("face not found")
            else:
                for face in faces:
                    (x, y, w, h) = face
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                    text_x = x + 10  # x坐标加上一些偏移
                    text_y = y + h + 20  # y坐标是矩形底部加上一些偏移
                    # 绘制文本
                    cv2.putText(frame, 'face', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('frame', frame)  # 显示读取到的这一帧画面，包含人脸边界框

        key = cv2.waitKey(25)  # 等待一段时间，并且检测键盘输入

        if key == ord('q'):  # 若是键盘输入'q',则退出，释放视频
            cap.release()
            break
    else:
        cap.release()

cap.release()
cv2.destroyAllWindows()