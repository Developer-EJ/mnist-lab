# -*- coding: utf-8 -*-
"""
MNIST 분류용 신경망 조립 모듈.

개별 layer를 OrderedDict에 쌓아 forward/backward 순서를 명확히 유지합니다.
"""

from collections import OrderedDict

import numpy as np

from activations import ReLU, Softmax
from layers import Affine, BatchNorm, Dropout
from losses import cross_entropy_loss


class NeuralNetwork:
    """
    MNIST 분류용 신경망.
    입력 784 -> 은닉층(들) -> 출력 10 (Softmax).
    은닉층 구성: Affine -> BatchNorm -> ReLU -> Dropout (모두 필수)
    가중치 초기화: He 또는 Xavier 중 선택.
    """

    def __init__(self, use_batchnorm=True, use_dropout=True, dropout_ratio=0.2):
        """
        Args:
            use_batchnorm: 은닉층마다 BatchNorm을 넣을지 여부
            use_dropout: 은닉층마다 Dropout을 넣을지 여부
            dropout_ratio: Dropout에서 끌 뉴런 비율
        """
        # TODO: params dict를 만들고 Affine/BatchNorm/ReLU/Dropout layer를 순서대로 구성하세요.
        # 권장 구조: 784 -> 512 -> 256 -> 10
        # self.layers는 OrderedDict로 만들고, self.grads는 params와 같은 key를 갖게 합니다.
        hidden_sizes = [512, 256]
        layer_sizes = [784] + hidden_sizes + [10]

        # He 초기화로 가중치 생성
        self.params = {}
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            self.params[f"W{i+1}"] = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
            self.params[f"b{i+1}"] = np.zeros(n_out)

        if use_batchnorm:
            for i in range(len(hidden_sizes)):
                self.params[f"gamma{i+1}"] = np.ones(hidden_sizes[i])
                self.params[f"beta{i+1}"] = np.zeros(hidden_sizes[i])

        # 은닉층: Affine -> (BatchNorm) -> ReLU -> (Dropout)
        self.layers = OrderedDict()
        for i in range(len(hidden_sizes)):
            self.layers[f"Affine{i+1}"] = Affine(
                self.params[f"W{i+1}"], self.params[f"b{i+1}"]
            )
            if use_batchnorm:
                self.layers[f"BatchNorm{i+1}"] = BatchNorm(
                    self.params[f"gamma{i+1}"], self.params[f"beta{i+1}"]
                )
            self.layers[f"ReLU{i+1}"] = ReLU()
            if use_dropout:
                self.layers[f"Dropout{i+1}"] = Dropout(dropout_ratio)

        # 출력층
        last = len(hidden_sizes) + 1
        self.layers[f"Affine{last}"] = Affine(
            self.params[f"W{last}"], self.params[f"b{last}"]
        )

        self.softmax = Softmax()
        self.use_batchnorm = use_batchnorm
        self.grads = {key: np.zeros_like(val) for key, val in self.params.items()}

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, 784) 정규화된 MNIST 이미지
            train: BatchNorm/Dropout의 학습 모드 여부

        Returns:
            (batch_size, 10) 각 숫자 클래스의 확률
        """
        # TODO: self.layers를 순서대로 통과시키고 마지막에 Softmax를 적용하세요.
        for layer in self.layers.values():
            if isinstance(layer, (BatchNorm, Dropout)):
                x = layer.forward(x, train)
            else:
                x = layer.forward(x)
        return self.softmax.forward(x)

    def backward(self, dout):
        """
        네트워크 전체 역전파를 수행하고 self.grads를 채웁니다.

        Args:
            dout: Softmax+CrossEntropy를 합친 출력층 gradient
        """
        # TODO: layer를 역순으로 통과시키고 Affine/BatchNorm의 gradient를 self.grads에 모으세요.
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        affine_idx = bn_idx = 1
        for layer in self.layers.values():
            if isinstance(layer, Affine):
                self.grads[f"W{affine_idx}"] = layer.dW
                self.grads[f"b{affine_idx}"] = layer.db
                affine_idx += 1
            elif isinstance(layer, BatchNorm):
                self.grads[f"gamma{bn_idx}"] = layer.dgamma
                self.grads[f"beta{bn_idx}"] = layer.dbeta
                bn_idx += 1

    def loss(self, x, y):
        """현재 모델의 예측 확률을 만든 뒤 cross entropy loss를 반환합니다."""
        y_pred = self.forward(x, train=True)
        return cross_entropy_loss(y_pred, y)

    def predict(self, x):
        """추론 모드로 확률을 예측합니다. BatchNorm/Dropout은 train=False로 동작합니다."""
        return self.forward(x, train=False)
