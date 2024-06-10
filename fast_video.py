# 导入必要的包
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
FaceCascade = cv2.CascadeClassifier('./moudel/haarcascade_frontalface_default.xml')


# 构造参数解析并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,help="path to input video file")
args = vars(ap.parse_args())
# 启动文件视频流线程并允许缓冲区开始填充
print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)
# 启动 FPS 计时器
fps = FPS().start()

# 循环播放视频文件流中的帧
while fvs.more():
	# 从线程视频文件流中抓取帧，调整大小，并将其转换为灰度（同时仍保留 3 个通道）
	frame = fvs.read()
	frame = imutils.resize(frame, width=450)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	frame = np.dstack([frame, frame, frame])

	# 在frame上显示队列的大小
	cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	# 显示帧并更新 FPS 计数器
	cv2.imshow("Frame", frame)
	cv2.waitKey(1)
	fps.update()

# 停止计时器并显示 FPS 信息
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# 做一些清理工作
cv2.destroyAllWindows()
fvs.stop()
