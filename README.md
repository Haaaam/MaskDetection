# MaskDetection

건물 출입 시 사람들의 마스크 착용여부 판단
-Face mask detector video

• 실시간 video stream을 활용하여 마스크 착용여부 확인

-	OpenCV 및 딥러닝으로 이미지영상에서 얼굴 감지
-	얼굴을 감지한 후, 각 얼굴 ROI에 얼굴 마스크 분류기를 적용
-	frame, faceNet(이지미에서 얼굴 위치를 감지하는 데 사용되는 모델), maskNet(Face Mask Detector 모델)의 세가지 함수 파라미터 사용 
-	Python,OpenCV 및 Keras/Tensorflow를 사용하여 face mask detector Video구현


![Mask Detection 개발 과정](https://user-images.githubusercontent.com/42646583/114132285-07ee9e00-993f-11eb-888e-8c0409718134.JPG)
