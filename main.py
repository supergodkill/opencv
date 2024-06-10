import cv2

#载入脸部模型
FaceCascade =cv2.CascadeClassifier('./moudel/haarcascade_frontalface_default.xml')
#载入眼部模型
FaceCascade_eye = cv2.CascadeClassifier('./moudel/haarcascade_eye.xml')
#载入嘴部模型
FaceCascade_mouse =cv2.CascadeClassifier('./moudel/haarcascade_smile.xml')



#图片路径
img_path="./testjpg/shabi.jpg"

#读取图片路径的图片
img = cv2.imread(img_path)

#判断照片是否正确读取
if img is None:
    print("Can't open the image")
    exit(1)
else:
    print("Image opened")
    #灰度转换
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print("Pic To Gray sucessful")

#检测人脸
faces  =  FaceCascade.detectMultiScale(gray,1.1,30)

eyes   =  FaceCascade_eye.detectMultiScale(gray,1.1,50)

mouses =  FaceCascade_mouse.detectMultiScale(gray,1.1,700)



if faces is  None :
    print("no face bro")
    exit(99)

else:
    print("have face ")
    # 遍历返回的face数组
    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        text_x = x + 10  # x坐标加上一些偏移
        text_y = y + h + 20  # y坐标是矩形底部加上一些偏移
        # 绘制文本
        cv2.putText(img, 'face', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print("face has beed drawd")

    for eye in eyes:
        (x, y, w, h) = eye
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
        text_x = x + 10  # x坐标加上一些偏移
        text_y = y + h + 20  # y坐标是矩形底部加上一些偏移
        # 绘制文本
        cv2.putText(img, 'eyes', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print("eyes has beed drawd")
    for mouse in mouses:
        (x, y, w, h) = mouse
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
        text_x = x + 10  # x坐标加上一些偏移
        text_y = y + h + 20  # y坐标是矩形底部加上一些偏移
        # 绘制文本
        cv2.putText(img, 'mouse', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print("mouse has beed drawd")


#调成图片大小
resizeimg=cv2.resize(img,(640,480))
#显示调整后的图片大小
cv2.imshow('fucking face',resizeimg)
#等待按键按下
cv2.waitKey(0)
#退出
cv2.destroyAllWindows()
