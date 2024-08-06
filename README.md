# 👦AutismProject 자폐 스펙트럼 진단👧
- 논문: [DeepASD: Facial Image Analysis for Autism Spectrum Diagnosis via Explainable Artificial Intelligence](https://ieeexplore.ieee.org/abstract/document/10200203)

## 프로젝트 배경
- 자폐스펙트럼은 개인에 따라 나타나는 증상이 다른 질환임
- 평생 영향을 끼치기 때문에 조기 중재가 중요하지만 진단 과정이 오래걸림
- 전문의의 개인적인 의견이 개입되기 때문에 진단을 위한 단일화된 도구가 필요함
- 자폐 스펙트럼이 발현하면 뇌 뿐만 아니라 얼굴에도 영향을 주어 보통 아이와 눈에 띄게 구별됨

## 프로젝트 목적
- 외모적 특성을 사용하여 자폐스펙트럼을 판단하고 Grad-CAM을 사용해 판단의 근거를 제시
- 진단 과정에서 시간을 줄이고 전문의의 판단에 도움이 되기 위한 단일도구로서의 역할

## 사용 데이터
- 데이터: Kaggle [Autism_Image_Data](https://www.kaggle.com/datasets/cihan063/autism-image-data)

## 주요 기능
- 데이터 전처리
    - 모든 이미지 Resize(224*224)
    - 0:정상, 1: 자폐 스펙트럼으로 이미지 분류
    - ImageGenerator를 사용한 이미지 증강
- CNN 모델
  - MobileNet
    
    ![MobileNet_구조도_회전](https://github.com/user-attachments/assets/770408ac-25d7-4e6f-805c-8e6052de8871)
  
  - Xception
    
   ![image](https://github.com/user-attachments/assets/43d76e28-413c-49d2-bb00-3c7cd93d77f4)
  
  - EfficientNet
  
    ![EfficientNet_구조도_회전](https://github.com/user-attachments/assets/1e5ff93d-e689-4c8d-98fb-32eab22da029)
  
  - Ensemble(MobileNet+Xception+EfficientNet)

    ![image](https://github.com/user-attachments/assets/c34959d3-1135-4306-a764-a8659f8a496d)
- 결과 시각화

  ![이미지 분류](https://github.com/user-attachments/assets/d02f1b7d-f6f5-44b5-a3db-ef7f9518b167)
## 결과
- 모든 모델에서 AUROC 0.8이상의 결과 도출((A)MobileNet: 0.80, (B)Xception:0.87, (C)EfficientNet:0.86, (D)Ensemble: 0.89)
  ![image](https://github.com/user-attachments/assets/4173cb11-6207-4c09-a4fc-dc6441942ae2)
  ![합친 AUroc](https://github.com/user-attachments/assets/1d77ccfa-1f58-4b01-8ec6-d457d09465ff)

