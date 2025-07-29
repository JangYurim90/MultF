import torch
import torch.nn as nn

class ChangeRateLoss(nn.Module):
    def __init__(self):
        super(ChangeRateLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        # 변화율 계산: (t+1)의 값과 (t)의 값 차이
        change_pred = y_pred[:, 1:] - y_pred[:, :-1]
        change_true = y_true[:, 1:] - y_true[:, :-1]
        
        # 변화율 차이에 대한 MSE 계산
        loss = torch.mean((change_pred - change_true) ** 2)
        return loss
    
# MAPE 손실 함수 구현
class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-8 
        # 절대 퍼센트 오차 계산
        loss = torch.abs((y_true - y_pred) / (y_true + epsilon))
        # MAPE 계산 (배치 평균)
        return torch.mean(loss) * 100  # 퍼센트로 변환
    

class ValueTrendAndGradientLoss(nn.Module):
    def __init__(self):
        super(ValueTrendAndGradientLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # 값 차이 (예측 값과 실제 값의 차이)
        value_loss = self.mse(y_pred, y_true)

        # 추세 방향성 (증가/감소 여부 비교)
        trend_pred = torch.sign(y_pred[:, 1:] - y_pred[:, :-1])
        trend_true = torch.sign(y_true[:, 1:] - y_true[:, :-1])
        trend_loss = torch.mean((trend_pred != trend_true).float())

        # 변화율 차이 (n번째와 n-1번째 값의 변화율 차이 비교)
        grad_pred = y_pred[:, 1:] - y_pred[:, :-1]
        grad_true = y_true[:, 1:] - y_true[:, :-1]
        gradient_loss = self.mse(grad_pred, grad_true)

        # 세 가지 손실을 결합 (가중치 조정 가능)
        total_loss = value_loss + 0.5 * trend_loss + 0.5 * gradient_loss
        return total_loss
