# MaskDetection

### 건물 출입 시 사람들의 마스크 착용여부 판단
-Face mask detector video
- dataset: https://github.com/imeamin/MaskDetectionDataset


### 개발 목적
- 최근 중국에서 발생한 COVID-19로 인해 전세계적으로 심각한 상황
- 바이러스의 확산을 막기위해서는 마스크, 손 소독 등의 주의 필요
- 하지만 여전히 길을 걷다 보면 빈번하게 보이는 마스크를 쓰지 않은 사람들
- 코로나 바이러스로 인해 전세계적으로 현재 누적 사망자 수는 447240명

•	열 감지기는 있지만 마스크 확인 장치 無
-	카페, pc방 등의 공공시설 이용 중 마스크를 미착용 하는 사람들로 인해 방역 복병
-	코로나 바이러스 확산을 막기위해 공공 시설 출입 전 방문자들의 체온 검사를 위한 열 감지 장치 有
-	하지만 마스크 착용 여부를 확인하기 위한 장치는 無  


• 실시간 video stream을 활용하여 마스크 착용여부 확인
※실시간 Video는 “detect_mask_video.py”로 파일 첨부하였습니다

-	OpenCV 및 딥러닝으로 이미지영상에서 얼굴 감지
-	얼굴을 감지한 후, 각 얼굴 ROI에 얼굴 마스크 분류기를 적용
-	frame, faceNet(이지미에서 얼굴 위치를 감지하는 데 사용되는 모델), maskNet(Face Mask Detector 모델)의 세가지 함수 파라미터 사용 
-	Python,OpenCV 및 Keras/Tensorflow를 사용하여 face mask detector Video구현

- with Mask
![image](https://user-images.githubusercontent.com/42646583/114133076-6cf6c380-9940-11eb-8a45-4b2d8ed2eefd.png)

- without Mask
![image](https://user-images.githubusercontent.com/42646583/114133486-1b026d80-9941-11eb-83e1-d6734ed2b650.png)




### MaskDetection 개발 과정
![Mask Detection 개발 과정](https://user-images.githubusercontent.com/42646583/114132285-07ee9e00-993f-11eb-888e-8c0409718134.JPG)

-  Face Mask Detector를 훈련시킨 것과 Face Mask Detector를 이미지/비디오에 적용하는 것으로 나누어 개발
- 위의 순서대로 개발과정을 거쳐 최종적으로 실시간으로 마스크 착용 유무를 확인하는 Face Mask Detector Video 개발

### Train Face Mask Detector
- 이미 구축된 open source의 dataset을 활용하여 추가적으로 학습을 시킴
- MobileNetV2를 사용하여 Transfer learning 수행(Fine tuning을 위해 사용)
            1. 사전에 훈련된 ImageNet 가중치로 MobileNet을 load
            2. 새로운 FC headModel을 구성하고 기존 head DB에 추가
            3. 네트워크의 기본 계층 고정시킴
- headModel=Dense(2,activation=”softmax”)(headModel)에서 입력받은 값을 0~1사이로 나타내는 softmax를 사용하여 마스크를 착용 [1,0]/마스크를 미착용 [0,1]이 나오도록   encoding 
- 이것을 cross entropy를 사용해서 학습(실제 값과 예측 값의 차 확인을 위함)  


• Face Mask Detector 훈련의 정확도 dataset 그래프 
- 저장된 영상에 마스크가 있는지 없는지를 train_mask_detector.py를 가지고 분석하고 훈련
- 출력된 결과 그래프를 보면 99%의 정확도를 얻는 것을 확인 가능
- 그래프에서 볼 수 있듯이 val_loss(검증 손실)이 train_loss(훈련 손실)보다 낮은 것을 확인 가능
- 훈련 과정을 거친 dataset을 가지고 테스트의 결과가 정확한지 확인
- 마스크를 착용했을 때와 착용하지 않은 상태를 정확하게 구분 가능

- Face Mask Detector 훈련 정확도 & 손실곡선
![image](https://user-images.githubusercontent.com/42646583/114133222-adeed800-9940-11eb-8aec-d86f2845d98e.png)


#### [References]
- Face Mask Dataset& Face detection
   - https://github.com/prajnasb/observations
   - https://arxiv.org/ftp/arxiv/papers/1207/1207.2922.pdf
- OpenCV 및 딥러닝 기반 face detection(실시간 video stream):https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector


