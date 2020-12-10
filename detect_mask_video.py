# USAGE
# python detect_mask_video.py


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# frame의 크기를 설정하고 blob을 만든다
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# 네트워크를 통해 blob을 통과시키고 face detection을 얻는다
	faceNet.setInput(blob)
	detections = faceNet.forward()

    #얼굴 목록 해당 위치 및 얼굴 마스크 네트워크의 예측 목록 초기화

	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):

		# detection과 관련된 신뢰도( 확률)을 추출
		confidence = detections[0, 0, i, 2]


		#확률이 최소 확률보다 큰지 확인하여 약한 detection을 제거
		if confidence > args["confidence"]:
			# 객체 bounding box의 (x,y)좌표 계산
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# bounding box가 프레임의 치수 내에 들어가도록 한다
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))


			# 얼굴 ROI를 추출하여 BGR에서 RGB 채널 순서로 변환한 후 크기를 224x224로 조정 후 전처리 수행
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# 면과 경계 상자를 해당 목록에 추가
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# 얼굴이 하나이상 감지된 경우에만 예측
	if len(faces) > 0:
		# 더 빠른 추론을 위해 for루프에서 하나씩 예측하는 것이 아니라 모든 면에 대하여 일괄 예측 수행
		preds = maskNet.predict(faces)

# 얼굴 위치와 해당 위치의 2-tuple 반환
	return (locs, preds)


# argument parser을 구성하고 argument를 파싱
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 디스크에서 직렬화된 face detector 모델 로드
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# 디스크에서 face mask detector 모델 로드
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# 비디오 스트림을 초기화하고 카메라 센서가 예열 되도록 함
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 비디오 스트림에서 프레임을 반복
while True:
	#스레드된 비드오 스트림에서 프레임을 잡고 최대 400pixel 너비로 크기를 조정
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# 프레임의 얼굴을 감지하고 마스크를 착용했는지 확인
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# 감지된 얼굴 위치와 해당 위치를 반복
	for (box, pred) in zip(locs, preds):
		# bounding box와 예측의 압축을 푼다
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# boundingbox와 text를 그리는데 사용항 클레스 레이블과 색상을 결정
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# label에 확률을 포함시킴
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# 출력 프레임에 label 및 bounding box 사각형을 표시
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# output frame 보여줌
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# 'q'누르면 비디오 종료
	if key == ord("q"):
		break


cv2.destroyAllWindows()
vs.stop()