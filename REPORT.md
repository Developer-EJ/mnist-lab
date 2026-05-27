# MNIST 손글씨 인식 과제 보고서

## 0. 반·팀원


| 항목     | 내용            |
| ------ | ------------- |
| **조**  | 301 - 5조    |
| **팀원** | 고명석, 강지현, 김원우, 김은재 |


---

## 1. 실험 목적

MNIST 10-class 분류를 **NumPy만으로 구현한 신경망**으로 수행하고, 테스트 정확도와 학습 과정을 보고합니다.

---

## 2. 모델 구조


| 구분      | 내용                                                               |
| ------- | ---------------------------------------------------------------- |
| **입력**  | 784 (28×28 픽셀, 0~1 정규화)                                          |
| **은닉층** | Affine → BatchNorm → ReLU → Dropout 순으로 구성 |
| **출력**  | Affine(→10) + Softmax                                            |


**모델 구조 : 2층 은닉**  
입력 784 → Affine(512) → BatchNorm → ReLU → Dropout → Affine(256) → BatchNorm → ReLU → Dropout → Affine(10) → Softmax

---

## 3. 학습 설정


| 항목                 | 값           |
| ------------------ | ----------- |
| 옵티마이저              | Adam        |
| 학습률 (lr)           | 0.001       |
| epochs             | 20          |
| batch_size         | 128         |
| Dropout 비율         | 0.5         |
| BatchNorm momentum | 0.9         |
| 가중치 초기화            | He (bias 0) |


---

## 4. 실험 환경

- Python 3.11, NumPy, Matplotlib
- 학습 소요 시간: CPU 기준 약 200초

---

## 5. 결과


| 항목           | 값              |
| ------------ | -------------- |
| **테스트 정확도**  | 98.45%    |
| **총 파라미터 수** | 537,354 |


### 손실 커브

- 학습 곡선: (그래프 이미지를 붙이거나, 예: "Epoch 1 Loss 0.42 → Epoch 20 Loss 0.06 수렴" 같이 수치로 요약)


## Batch Size Compare

| Model | Train Loss | Test Loss | Accuracy | Total Params | Time |
|---|---:|---:|---:|---:|---:|
| batch_10 | 0.0653 | 0.0490 | 98.49% | 537,354 | 1790.33s |
| batch_128 | 0.0135 | 0.0607 | 98.38% | 537,354 | 209.88s |
| batch_60000 | 0.3279 | 0.2548 | 92.50% | 537,354 | 106.06s |

---
![alt text](image.png)

## Learning Rate Compare

| Model | Train Loss | Test Loss | Accuracy | Total Params | Time |
|---|---:|---:|---:|---:|---:|
| lr_100 | 13.3510 | 13.3918 | 9.58% | 537,354 | 221.26s |
| lr_0.1 | 0.0975 | 0.0961 | 97.89% | 537,354 | 213.79s |
| lr_0.001 | 0.0149 | 0.0537 | 98.44% | 537,354 | 203.42s |
| lr_0.00001 | 0.2085 | 0.1600 | 95.31% | 537,354 | 205.01s |

---
![alt text](image-1.png)


## Optimizer Compare
| Model | Train Loss | Test Loss | Accuracy | Total Params | Time |
|---|---:|---:|---:|---:|---:|
| optimizer_adam | 0.0142 | 0.0598 | 98.45% | 537,354 | 216.86s |
| optimizer_sgd | 0.3858 | 0.2930 | 92.11% | 537,354 | 163.61s |

![alt text](image-2.png)


![alt text](image-4.png)


![alt text](image-3.png)

dropout_off  | best_epoch=  8 | best_test_loss=0.2814 | final_test_loss=0.3774 | final_train_loss=0.0001 | train_acc=100.00% | test_acc=92.40%
dropout_0.1  | best_epoch=  8 | best_test_loss=0.2854 | final_test_loss=0.3922 | final_train_loss=0.0001 | train_acc=100.00% | test_acc=92.45%
dropout_0.2  | best_epoch=  9 | best_test_loss=0.2720 | final_test_loss=0.3822 | final_train_loss=0.0003 | train_acc=100.00% | test_acc=92.68%
dropout_0.5  | best_epoch= 20 | best_test_loss=0.2689 | final_test_loss=0.3414 | final_train_loss=0.0032 | train_acc=100.00% | test_acc=92.63%
dropout_0.9  | best_epoch=100 | best_test_loss=0.3950 | final_test_loss=0.3950 | final_train_loss=0.5129 | train_acc=94.00% | test_acc=88.48%
## 6. 회고

- 손실 수렴 여부, 과적합/과소적합 여부
- 구조·학습률·Dropout 등 변경 시도와 그 결과 (있다면 간단히)

