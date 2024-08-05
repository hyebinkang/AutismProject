# AutismProject 자폐 스펙트럼 진단
- 논문:DeepASD: Facial Image Analysis for Autism Spectrum Diagnosis via Explainable Artificial Intelligence
- <link>https://ieeexplore.ieee.org/abstract/document/10200203</link>

## 프로젝트 배경
- 자폐스펙트럼은 개인에 따라 나타나는 증상이 다른 질환임
- 평생 영향을 끼치기 때문에 조기 중재가 중요하지만 진단 과정이 오래걸림
- 전문의의 개인적인 의견이 개입되기 때문에 진단을 위한 단일화된 도구가 필요함
- 자폐 스펙트럼이 발현하면 뇌 뿐만 아니라 얼굴에도 영향을 주어 보통 아이와 눈에 띄게 구별됨

## 프로젝트 목적
- 외모적 특성을 사용하여 자폐스펙트럼을 판단하고 Grad-CAM을 사용해 판단의 근거를 제시
- 진단 과정에서 시간을 줄이고 전문의의 판단에 도움이 되기 위한 단일도구로서의 역할

## 사용 데이터
- 데이터: 캐글

## 주요 기능
- 데이터 전처리
    - 모든 이미지 Resize(224*224)
    - 0:정상, 1: 자폐 스펙트럼으로 이미지 분류
    - ImageGenerator를 사용한 이미지 증강
- CNN 모델
  - MobileNet
  - Xception
  - EfficientNet
  - Ensemble(MobileNet+Xception+EfficientNet)
- 결과 시각화

## 결과
- 모든 모델에서 AUROC 0.8이상의 결과 도출(MobileNet: 0.80, Xception:0.87, EfficientNet:0.86, Ensemble: 0.89)
