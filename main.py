#!/usr/bin/env python3

import numpy as np
import pandas as pd
from operator import itemgetter
import pandas as pd
from matplotlib import pyplot, patches,axes
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from glob import glob
import seaborn as sns
import scipy
import torch.utils
from torch.utils.data import DataLoader, Dataset, Sampler
from argument import args_parser
from dataloader import data_split, windowDataset, hcp_data_load, subwindowDataset, SortedWindowDataset, BatchWindowDataset, WindowDataset, SortedBatchWindowDataset
import torch
from torch import nn
from models_encoder.layers import TSTransformerEncoder
from tqdm import tqdm
from datetime import datetime
#import xgboost as xgb
from models_encoder.loss import ChangeRateLoss, ValueTrendAndGradientLoss
from torch.nn.parallel import DistributedDataParallel as DDP
#from lightgbm import LGBMRegressor
from sklearn.svm import SVR
import time
import psutil
import torch
import torch.nn as nn



class SubjectSampler(Sampler):
    def __init__(self, dataset):
        self.subject_ids = dataset.subject_ids

    def __iter__(self):
        # subject_ids 순서대로 index를 반환
        indices = np.arange(len(self.subject_ids))
        return iter(indices)

    def __len__(self):
        return len(self.subject_ids)
    
    

class ValueAndTrendLoss(nn.Module):
    def __init__(self):
        super(ValueAndTrendLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # 값 차이 (예측과 실제 값의 차이)
        value_loss = self.mse(y_pred, y_true)
        
        # 추세 (n번째와 n-1번째 값의 차이가 증가했는지 감소했는지 비교)
        trend_pred = torch.sign(y_pred[:, 1:] - y_pred[:, :-1])
        trend_true = torch.sign(y_true[:, 1:] - y_true[:, :-1])

        # 추세 일치율 (같은 방향으로 변하는지 여부 확인)
        trend_loss = torch.mean((trend_pred != trend_true).float())

        # 두 손실을 결합 (가중치 조정 가능)
        total_loss = value_loss * 0.8+ 0.2 * trend_loss
        return total_loss

class ChangeRateLoss(nn.Module):
    def __init__(self, mse_weight=1.0, change_rate_weight=1.0, trend_penalty_weight=0.1):
        """
        :param mse_weight: 기본 MSE 손실 가중치
        :param change_rate_weight: 변화율 기반 손실 가중치
        :param trend_penalty_weight: 추세 패널티 가중치
        """
        super(ChangeRateLoss, self).__init__()
        self.mse_loss = nn.MSELoss() # 기본 MSE 손실 함수
        self.mae_loss = nn.L1Loss()  # MAE 손실 함수
        self.mse_weight = mse_weight  # MSE 가중치
        self.change_rate_weight = change_rate_weight  # 변화율 손실 가중치
        self.trend_penalty_weight = trend_penalty_weight  # 추세 패널티 가중치

    def forward(self, y_pred, y_true):
        # 1. 기본 MSE 손실
        mse_loss = self.mse_loss(y_pred, y_true)
        mae_loss = self.mae_loss(y_pred, y_true)

        # 2. 변화율 차이에 대한 손실 (MSE)
        pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        true_diff = y_true[:, 1:] - y_true[:, :-1]
        change_rate_loss = torch.mean((pred_diff - true_diff) ** 2)

        # 3. 추세 패널티: 실제 값은 증가하지만 예측 값이 감소하는 경우, 또는 그 반대의 경우 패널티 부여
        trend_penalty_mask = ((true_diff > 0) & (pred_diff < 0)) | ((true_diff < 0) & (pred_diff > 0))
        trend_penalty = torch.sum(torch.abs(pred_diff[trend_penalty_mask]))  # 패널티 절대값 합계

        ## L2 정규화 추가
        l2_lambda = 0.01
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        
        total_loss = (self.mse_weight * mse_loss + 
                    self.change_rate_weight * change_rate_loss + 
                    self.trend_penalty_weight * trend_penalty +
                    l2_lambda * l2_norm)
        
        return total_loss


    
class loss_combine(nn.Module):
    def __init__(self, lambdad_rate = 0.1):
        super(loss_combine, self).__init__()
        self.loss1 = nn.MSELoss()
        self.loss2 = ChangeRateLoss()
        self.MAEloss = nn.L1Loss()
        self.smoothloss = nn.SmoothL1Loss()
        self.lambdad_rate = lambdad_rate
        
    def forward(self,  y_pred, y_true):
        mse_loss = self.loss1(y_pred, y_true)
        mae_loss = self.MAEloss(y_pred, y_true)
        smooth_l1_loss_fn = self.smoothloss(y_pred, y_true)
        
        change_rate_loss = self.loss2(y_pred, y_true)
        loss =  mse_loss  * (1-self.lambdad_rate) + change_rate_loss * self.lambdad_rate
        return loss
        


# 경로가 없으면 생성하는 함수
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_code_as_txt(source_file, destination_path):
    try:
        # source_file: 원본 파일 경로 (e.g., "argument.py")
        # destination_path: 저장할 txt 파일 경로 (e.g., "/path/to/destination/output.txt")
        
        # 1. 원본 Python 파일 읽기
        with open(source_file, 'r') as source:
            code = source.read()
        
        # 2. 읽은 코드를 txt 파일로 저장
        with open(destination_path, 'w') as destination:
            destination.write(code)
        
        print(f"Code has been successfully saved to {destination_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# 현재 시간을 "년-월-일_시-분-초" 형식으로 반환하는 함수
def get_time_string():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def generate_padding_mask(batch_size, seq_length):
    # (batch_size, seq_length) 크기의 패딩 마스크를 생성
    # 패딩 위치는 0, 유효한 데이터 위치는 1
    mask = torch.ones(batch_size, seq_length)  # 기본적으로 모든 위치를 1로 설정
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_padding_mask_bool(batch_size, seq_length):
    # 기본적으로 모든 위치를 1로 설정 (유효한 위치)
    mask = torch.ones(batch_size,seq_length, dtype=torch.bool)  # Boolean 타입으로 설정
    return mask


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def train(model, train_loader,i, ML = 'None', ML_model = None):
    batchloss = 0.0
    print("Epoch: ", i)
    model.to(device)
    model = nn.DataParallel(model, device_ids=[0,1])
    iter_count = 0
    time_now = time.time()


    for i_e, (inputs, outputs) in enumerate(train_loader):
        iter_count += 1
        optimizer.zero_grad(set_to_none=True)
        
        # Inputs to encoder (src), and targets for decoder (tgt)
        #inputs = inputs.transpose(0, 1)

        src = inputs.float().to(device)  # Input timepoints (e.g., 100 timepoints)
        
        #outputs = outputs.transpose(0, 1)

        #tgt = outputs[:, :-1, :].float().to(device)  # Previous timepoints for decoder (teacher forcing)
        #target_output = outputs[:, 1:, :].float().to(device)  # Next 10 timepoints
        tgt = outputs.float().to(device)
        

        src_mask = generate_padding_mask_bool(batch_size=src.shape[0],seq_length=src.shape[1]).to(device)
        #tgt_mask = generate_padding_mask(batch_size=args.batch_size,seq_length=tgt.shape[0]).to(device)
        
        if ML == 'xgb':

            
            # Get the output and Q, K, V from transformer encoder
            result, all_q, all_k,all_v = model(src, src_mask)
            result_np = result.detach().cpu().numpy()
            result_np = result_np.reshape(-1, result_np.shape[-1])
            
            # 만약 XGBoost 모델이 학습되어 있지 않다면 학습 (훈련)
            outputs_np = outputs.detach().cpu().numpy()
            outputs_np = outputs_np.reshape(-1, outputs_np.shape[-1])
            print(result_np.shape, outputs_np.shape)
            
            result_np = result_np.flatten()
            outputs_np = outputs_np.flatten()
            
            print(result_np.shape, outputs_np.shape)
            
            ML_model.fit(result_np, outputs_np)
            
            # 학습된 XGBoost 모델을 통해 예측 수행
            ML_preds = ML_model.predict(result_np)

            # XGBoost 예측을 PyTorch 텐서로 변환하여 트랜스포머 결과와 결합
            ML_preds_tensor = torch.tensor(ML_preds, device=device).view(result.shape)
            
            combined_result = ML_preds_tensor.to(device)
            outputs = outputs.to(device) 
            
            if combined_result.shape == outputs.shape:
                loss = criterion(combined_result, outputs.float())
            else:
                print("Shape mismatch:", combined_result.shape, outputs.shape)
                continue
            
        elif ML == 'svms':
            result, all_q, all_k,all_v,svr_model = model(src, src_mask,ML=args.ML,svr_model = ML_model)
                
            
        else: 
            # Get the output and Q, K, V from transformer encoder
            result, all_q, all_k,all_v = model(src, src_mask,ML = args.ML)
            
            # 수정 후: 크기가 맞는지 확인하고 loss 계산
            result = result.to(device)  # result를 device로 이동
            outputs = outputs.to(device)  # outputs을 device로 이동
            

            # 배치 크기와 시퀀스 길이가 일치하는지 확인
            if result.shape == outputs.shape:
                loss = criterion(result, outputs.float())
            else:
                print("Shape mismatch:", result.shape, outputs.shape)

            loss = criterion(result, outputs.float().to(device))

        #batch_loss = torch.sum(loss)
        mean_loss = torch.mean(loss)  # mean loss (over active elements) used for optimization
        
        
        total_loss = mean_loss
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        batchloss += loss.item()

        if (i_e+1) % 100==0:
            cpu_pct = psutil.cpu_percent(interval=None)
            ram_pct = psutil.virtual_memory().percent

            if torch.cuda.is_available():
                gpu_alloc = torch.cuda.memory_allocated() / 1024**2
                gpu_resv  = torch.cuda.memory_reserved()  / 1024**2
                msg = (f"[Iter {i:4d}] CPU {cpu_pct:5.1f}% / RAM {ram_pct:5.1f}% | "
                    f"GPU alloc {gpu_alloc:6.1f} MiB / reserved {gpu_resv:6.1f} MiB")
            else:
                msg = f"[Iter {i:4d}] CPU {cpu_pct:5.1f}% / RAM {ram_pct:5.1f}%"

            print(msg)
            log_file.write(msg + '\n')
            log_file.flush()
            
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i_e + 1, i + 1, loss.item()))
            speed = (time.time()-time_now)/iter_count
            left_time = speed*((args.epoch - i)*train_steps - i_e)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()
    
    if len(train_loader) > 0:
        progress.set_description("{:0.5f}".format(batchloss / len(train_loader)))
        print(f"lr: {scheduler.get_last_lr()[0]}")
        train_losses.append(batchloss / len(train_loader))
    else:
        progress.set_description("No data in train_loader")
        
    return train_losses, ML_model

def validation(model, val_loader,ML = 'None', ML_model = None):
    model.to(device)
    model = nn.DataParallel(model, device_ids=[0,1])
    
    model.eval()
    total_val_loss = 0.0
    

    with torch.no_grad():
        for (inputs, outputs) in val_loader:
            #inputs = inputs.transpose(0, 1)
            src = inputs.float().to(device)

            #outputs = outputs.transpose(0, 1)
            tgt = outputs.float().to(device)

            src_mask = generate_padding_mask_bool(batch_size=src.shape[0],seq_length=src.shape[1]).to(device)
            tgt_mask = generate_padding_mask(batch_size=args.batch_size,seq_length=tgt.shape[0]).to(device)
            
            if ML == 'xgb':
                # Get the output and Q, K, V from transformer encoder
                result, all_q, all_k,all_v = model(src, src_mask)
                result_np = result.detach().cpu().numpy()
                result_np = result_np.reshape(-1, result_np.shape[-1])
                
                ML_preds = ML_model.predict(result_np)
                ML_preds_tensor = torch.tensor(ML_preds, device=device,requires_grad=False).view(result.shape)
                
                result = ML_preds_tensor.to(device)
                outputs = outputs.to(device) 

            else:
                # Get the output and Q, K, V from both encoder and decoder
                result, all_q, all_k,all_v = model(src, src_mask,ML = args.ML)
                outputs = outputs.to(device)

            val_loss = criterion(result, outputs.float().to(device))
            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(val_loader)
    # 초록색으로 텍스트 출력
    print(f"Validation Loss: {average_val_loss}")
    
    return average_val_loss
    
def test(model, test_loader,model_path,ml_path, ML = 'None'):
    print("Testing...\n\n")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    total_test_loss = 0.0
    loss_per_subject = []
    plot = {'result': [], 'output': []}
    
    with torch.no_grad():
        num_samples = 0
        num_samples_loss = 0.0
        
        for (inputs, outputs,sub_idx) in test_loader:
            #inputs = inputs.transpose(0, 1)
            src = inputs.float().to(device)

            #outputs = outputs.transpose(0, 1)
            tgt = outputs.float().to(device)

            src_mask = generate_padding_mask_bool(batch_size=src.shape[0],seq_length=src.shape[1]).to(device)
            tgt_mask = generate_padding_mask(batch_size=args.batch_size,seq_length=tgt.shape[0]).to(device)
            
            if ML == 'xgb':
                #ML_model_test = xgb.XGBRegressor(objective='reg:squarederror')
                ML_model_test = LGBMRegressor()
                ML_model_test.load_model(model_path)
                
                result, all_q, all_k,all_v = model(src, src_mask)
                result_np = result.detach().cpu().numpy()
                result_np = result_np.reshape(-1, result_np.shape[-1])
                
                ML_preds = ML_model_test.predict(result_np)
                ML_preds_tensor = torch.tensor(ML_preds, device=device).view(result.shape)
                result = ML_preds_tensor.to(device)
                
            else:
                # Get the output and Q, K, V from both encoder and decoder
                result, all_q, all_k,all_v = model(src, src_mask,ML = args.ML)

                outputs = outputs.to(device)

            test_loss = criterion(result, outputs.float().to(device))
            total_test_loss += test_loss.item()
            
            plot['result'].append(result.cpu().numpy())
            plot['output'].append(outputs.cpu().numpy())    
            
            num_samples += inputs.shape[1]
            num_samples_loss += test_loss.item()
            if num_samples % 470 == 0:
                print(f"Test Loss: {num_samples_loss / 470}")
                loss_per_subject.append(num_samples_loss / 470)
                num_samples_loss = 0.0
                print("{}th_subject  done".format(num_samples/470))
                
        print(f"Test Loss: {total_test_loss / len(test_loader)}")
        average_test_loss = total_test_loss / len(test_loader)
        
        return average_test_loss, loss_per_subject, plot


# Future prediciton
def train_future(model, train_loader, i, ML='None', ML_model=None):
    batchloss = 0.0
    print("Epoch: ", i)
    init = 0
    previous_subject_id = None

    for batch_idx, (inputs, outputs, subject_ids) in enumerate(train_loader):
        current_subject_id = subject_ids[0]

        # 새로운 서브젝트 감지 시 초기화
        if previous_subject_id is None or current_subject_id != previous_subject_id:
            #print(f"New subject {current_subject_id} detected. Resetting initial window.")
            init = 0

        if init != 0:
            init_inputs[:, :-2, :] = init_inputs[:, 2:, :]  # 이전 입력 업데이트
            init_inputs[:, -2:, :] = result.detach()  # 이전 결과를 새로운 입력으로
            inputs = init_inputs

        # inputs와 outputs이 numpy 배열일 경우 Tensor로 변환
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs)
        if isinstance(outputs, np.ndarray):
            outputs = torch.tensor(outputs)

        # Tensor로 변환 후 이동
        src = inputs.float().to(device)
        tgt = outputs.float().to(device)

        # 마스크 생성
        src_mask = generate_padding_mask_bool(batch_size=src.shape[0], seq_length=src.shape[1]).to(device)
        tgt_mask = generate_padding_mask(batch_size=src.shape[0], seq_length=tgt.shape[0]).to(device)

        # Forward pass
        result, all_q, all_k, all_v = model(src, src_mask, ML=args.ML)
        # Result와 Outputs 디바이스 확인 및 이동
        outputs = outputs.to(device)
        #result = result.to('cpu')
        #tgt = tgt.float().to(device)  # outputs를 device로 이동

        # 초기화 입력 업데이트
        if init == 0:
            init_inputs = inputs.clone()

        previous_subject_id = current_subject_id
        init += 1

        # Loss 계산
        if result.shape == outputs.shape:
            loss = criterion(result.float(), outputs.float())
        else:
            print("Shape mismatch:", result.shape, outputs.shape)
            continue

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # Gradients 초기화

        # 그래프와 분리된 손실 값 누적
        batchloss += loss.item()

    # 평균 손실 계산 및 저장
    if len(train_loader) > 0:
        progress.set_description("{:0.5f}".format(batchloss / len(train_loader)))
        train_losses.append(batchloss / len(train_loader))
    else:
        progress.set_description("No data in train_loader")

    return train_losses, ML_model

def validation_future(model, val_loader,ML = 'None', ML_model = None):
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        init = 0
        previous_subject_id = None
        for batch_idx,(inputs, outputs, subject_ids) in enumerate(test_future_loader):
            current_subject_id = subject_ids[0]
            
            if previous_subject_id is None or current_subject_id != previous_subject_id:
                #print(f"New subject {current_subject_id} detected. Resetting initial window.")
                init = 0
            
                
            if init !=0:
                init_inputs[:,:-2,:]=init_inputs[:,2:,:]
                init_inputs[:,-2:,:] = result
                inputs = init_inputs
                
            if isinstance(inputs, np.ndarray):
                inputs = torch.tensor(inputs)
            if isinstance(outputs, np.ndarray):
                outputs = torch.tensor(outputs)
                
            src = inputs.float().to(device)

            #outputs = outputs.transpose(0, 1)
            tgt = outputs.float().to(device)

            src_mask = generate_padding_mask_bool(batch_size=src.shape[0],seq_length=src.shape[1]).to(device)
            tgt_mask = generate_padding_mask(batch_size=args.batch_size,seq_length=tgt.shape[0]).to(device)
            
            
            # Get the output and Q, K, V from both encoder and decoder
            result, all_q, all_k,all_v = model(src, src_mask,ML = args.ML)
            outputs = outputs.to(device)
            init_inputs = inputs
            previous_subject_id = current_subject_id
                
            init = init +1

            val_loss = criterion(result, outputs.float().to(device))
            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(val_loader)
    # 초록색으로 텍스트 출력
    print(f"Validation Loss: {average_val_loss}")
    
    return average_val_loss


def test_future(model, test_loader,model_path,ml_path, ML = 'None'):
    print("Testing...\n\n")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    total_test_loss = 0.0
    loss_per_subject = []
    plot = {'result': [], 'output': []}
    
    with torch.no_grad():
        num_samples = 0
        num_samples_loss = 0.0
        init = 0
        previous_subject_id = None
        for batch_idx,(inputs, outputs, subject_ids) in enumerate(test_loader):
            current_subject_id = subject_ids[0]
            
            if previous_subject_id is None or current_subject_id != previous_subject_id:
                print(f"New subject {current_subject_id} detected. Resetting initial window.")
                init = 0
            
                
            if init !=0:
                init_inputs[:,:-args.o_win,:]=init_inputs[:,args.o_win:,:]
                init_inputs[:,-args.o_win:,:] = result
                inputs = init_inputs
                
            if isinstance(inputs, np.ndarray):
                inputs = torch.tensor(inputs)
            if isinstance(outputs, np.ndarray):
                outputs = torch.tensor(outputs)
                
            src = inputs.float().to(device)

            #outputs = outputs.transpose(0, 1)
            tgt = outputs.float().to(device)

            src_mask = generate_padding_mask_bool(batch_size=src.shape[0],seq_length=src.shape[1]).to(device)
            tgt_mask = generate_padding_mask(batch_size=args.batch_size,seq_length=tgt.shape[0]).to(device)
            
                
            
            # Get the output and Q, K, V from both encoder and decoder
            result, all_q, all_k,all_v = model(src, src_mask,ML = args.ML)

            outputs = outputs.to(device)
            init_inputs = inputs
            previous_subject_id = current_subject_id
                
            init = init +1
            #test_loss = criterion(result, outputs.float().to(device))
            #total_test_loss += test_loss.item()
            
            
            plot['result'].append(result.cpu().numpy())
            plot['output'].append(outputs.cpu().numpy())    
            
        #     num_samples += inputs.shape[1]
        #     num_samples_loss += test_loss.item()
        #     if num_samples % 470 == 0:
        #         print(f"Test Loss: {num_samples_loss / 470}")
        #         loss_per_subject.append(num_samples_loss / 470)
        #         num_samples_loss = 0.0
        #         print("{}th_subject  done".format(num_samples/470))
                
        # print(f"Test Loss: {total_test_loss / len(test_loader)}")
        # average_test_loss = total_test_loss / len(test_loader)
        average_test_loss = 0.0 
        
        return average_test_loss, loss_per_subject, plot
    
def test_future_sequential(model, test_loader, model_path, ML='None'):
    """
    한 서브젝트에 대해 순차적으로 한 윈도우씩 예측하는 방식.
    배치 크기는 2048로 유지하되, 실제 예측은 배치 내 한 샘플만 사용합니다.
    
    model_path: 모델 가중치 경로
    args.o_win: output window 길이 (예: 10)
    args.n_steps: 예측을 진행할 총 단계 수
    """
    print("Testing sequentially for one subject...\n")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    predictions = []  # 각 단계의 예측 결과 저장
    plot = {'result': [], 'output': []}
    loss_per_subject = []

    # test_loader는 서브젝트별로 정렬되어 있다고 가정.
    # 여기서는 첫 번째 배치에서 첫 윈도우를 가져옵니다.
    # 만약 배치 내 모든 샘플이 같은 서브젝트라면, 그 중 하나만 선택합니다.
    for batch_idx, (inputs, outputs, subject_ids) in enumerate(test_loader):
        # 배치 내 첫 번째 샘플만 사용한다고 가정
        current_subject = subject_ids[0]
        print(f"Processing subject {current_subject}")
        # current_input은 shape: (1, input_window, num_features)
        current_input = inputs[0:1].clone()  # 1개만 선택
        break  # 첫 배치만 사용

    # n_steps 만큼 순차적으로 예측 진행
    n_steps = 2375  # 예측할 단계 수 (예: 50)
    for step in range(n_steps):
        # 모델에 넣을 입력: current_input (shape: (1, input_window, num_features))
        src = current_input.float().to(device)
        src_mask = generate_padding_mask_bool(batch_size=src.shape[0], seq_length=src.shape[1]).to(device)
        
        with torch.no_grad():
            result, all_q, all_k, all_v = model(src, src_mask)
        
        # result의 shape는 (1, output_window, num_features)
        plot['result'].append(result.cpu().numpy())
        plot['output'].append(outputs.cpu().numpy())   
        
        # 업데이트: current_input를 시프트하고, 마지막 output_window 부분에 result를 추가
        # 예를 들어, 입력 윈도우 길이가 2048, output window 길이가 args.o_win라고 하면:
        # current_input[:, : -args.o_win, :] <- current_input[:, args.o_win:, :]
        # current_input[:, -args.o_win:, :] <- result
        current_input[:, : -args.o_win, :] = current_input[:, args.o_win:, :].clone()
        current_input[:, -args.o_win:, :] = result.clone()
        print(f"Step {step+1}/{n_steps} prediction done.")
        
    average_test_loss = 0.0 

    return average_test_loss, loss_per_subject, plot


    
def collate_fn(batch):
    # 데이터와 subject_id를 분리
    inputs, outputs, subject_ids = zip(*[(item[0], item[1], item[2]) for item in batch])

    # Inputs와 Outputs 개별적으로 Tensor로 변환
    processed_inputs = [torch.tensor(input_) for input_ in inputs]
    processed_outputs = [torch.tensor(output_) for output_ in outputs]

    # 크기 맞추기: torch.stack으로 배치 생성
    inputs = torch.stack(processed_inputs)
    outputs = torch.stack(processed_outputs)
    subject_ids = torch.tensor(subject_ids)
    return inputs, outputs , subject_ids

def print_subject_batch_info(dataloader):
    subject_batch_counts = defaultdict(int)  # 각 서브젝트의 배치 개수 저장
    batch_window_counts = []  # 각 배치의 윈도우 수 저장

    for batch_idx, (inputs, outputs, subject_ids) in enumerate(dataloader):
        unique_subjects = torch.unique(subject_ids).tolist()

        # 각 subject_id의 배치 개수 증가
        for subject_id in unique_subjects:
            subject_batch_counts[subject_id] += 1
        
        # 각 배치의 윈도우 수 저장
        batch_window_counts.append(inputs.shape[0])  # 배치 크기 (윈도우 수)

        print(f"Batch {batch_idx}: Subject IDs = {unique_subjects}, Number of windows = {inputs.shape[0]}")

    # 서브젝트별 배치 개수 출력
    print("\nSubject-wise batch counts:")
    for subject_id, batch_count in subject_batch_counts.items():
        print(f"Subject {subject_id}: {batch_count} batches")

    # 전체 배치 윈도우 수 출력
    print("\nBatch-wise window counts:")
    for batch_idx, window_count in enumerate(batch_window_counts):
        print(f"Batch {batch_idx}: {window_count} windows")
        
def lr_lambda(current_step):
    # warmup 단계: 현재 스텝이 warmup_steps보다 작으면 선형 증가
    if current_step < warmup_steps:
        return float(current_step) / float(max(1,warmup_steps))
    # 그 이후에는 예시로 선형 감소하도록 설정 (원하는 schedule로 수정 가능)
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

# 호출

   
args = args_parser()
device = torch.device("cuda:0")

# Save the model with the lowest validation loss
save_dir = "{}/A{}_i{}_o{}_lr{}_nl{}_sub{}_ep{}_{}_MSE_tanh".format(args.loss_dir, args.atlas, args.i_win, args.o_win, args.lr,args.nlayers,80,args.epoch,args.ML)
create_directory_if_not_exists(save_dir)

source_file = '/camin1/yrjang/Brain_network_dynamics/Dynamics_RNN/argument.py'
destination_path = f'{save_dir}/hyperparameter.txt'

save_code_as_txt(source_file, destination_path)

time_str = get_time_string()  

#load_data = np.load(args.HCPdata_dir)

# data만 따로 추출 하는 코드
load_data = hcp_data_load(args.atlas,args.HCPdata_dir)

if args.other_data != 'other_data':
    # data split
    train_data, val_data, test_data = data_split(load_data,args.train_size,args.val_size,args.test_size,random_state = 3)
else:
    test_data = load_data


# data_loader 
if args.mode == 'train':
    del test_data
    
    print('Start windowdataset')
    
    train_dataset = WindowDataset(train_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride, batch_size = args.batch_size)
    
    print("Mid windowdataset")
    val_dataset = WindowDataset(val_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride,batch_size = args.batch_size)
    
    print('End windowdataset')
    del train_data
    del val_data
    
    print('Start Dataloader')
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True)
    print("Mid Dataloader")
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,drop_last=True)
    print('End Dataloader')
    
    
    del train_dataset
    del val_dataset

elif args.mode == 'test':
    
    if args.other_data != 'other_data':
        del train_data
        del val_data
    
    #test_dataset = WindowDataset(test_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride , batch_size = args.batch_size)
    test_dataset = SortedBatchWindowDataset(test_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride , batch_size = args.batch_size)

    del test_data
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,drop_last=False)
    del test_dataset
    





# future prediction
if args.mode == 'train_future':
    train_future_dataset = SortedWindowDataset(train_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride, batch_size = args.batch_size)
    val_future_dataset = SortedWindowDataset(val_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride,batch_size = args.batch_size)
    
    train_future_loader = DataLoader(train_future_dataset,batch_size=args.batch_size,drop_last=True)
    val_future_loader = DataLoader(val_future_dataset,batch_size=args.batch_size,drop_last=True)
    
    
elif args.mode == 'test_future':
    test_future_dataset = SortedWindowDataset(test_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride , batch_size = args.batch_size)
    test_future_loader = DataLoader(test_future_dataset,shuffle=False,batch_size=args.batch_size,drop_last=False)


#print_subject_batch_info(test_future_loader)

# Load model (transformer)
# model = TFModel(d_model=args.d_model, nhead=args.nhead, nhid=args.nhid, nlayers=args.nlayers,input_length=args.i_win, output_length=args.o_win, dropout=args.dropout).to(device)
model = TSTransformerEncoder(feat_dim=args.feature_dim, max_len =args.i_win,d_model=args.d_model,n_heads=args.nhead,num_layers=args.nlayers,dim_feedforward=args.feature_dim ,linear_hidden=args.linear_hidden,output_length=args.o_win,dropout=args.dropout).to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
training_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(
    f"Total parameters: {total_params:,}, "
    f"Trainable parameters: {training_params:,}"
)


# # DataParallel로 모델 감싸기
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs")
#     model = nn.DataParallel(model)


#criterion = loss_combine(args.lambda_r)
#criterion= ChangeRateLoss(mse_weight=0.4, change_rate_weight=0.3, trend_penalty_weight=0.3)
#criterion = ValueAndTrendLoss()
#criterion = ValueTrendAndGradientLoss()
criterion = nn.MSELoss(reduction='mean')
#criterion = nn.L1Loss()

if args.mode == 'train':
    total_steps = len(train_loader) * args.epoch
    warmup_steps = len(train_loader) * args.warmup_steps

elif args.mode == 'test':
    total_steps = len(test_loader) * args.epoch
    warmup_steps = len(test_loader) * args.warmup_steps
    
elif args.mode == 'train_future':  
    total_steps = len(train_future_loader) * args.epoch
    warmup_steps = len(train_future_loader) * args.warmup_steps

elif args.mode == 'test_future':
    total_steps = len(test_future_loader) * args.epoch
    warmup_steps = len(test_future_loader) * args.warmup_steps

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)





# Train 및 Validation loss 저장할 리스트 초기화
train_losses = []
val_losses = []

# train
if args.mode == 'train':
    log_path = '/camin1/yrjang/Brain_network_dynamics/Dynamics_RNN/resource_usage.txt'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, 'a')  # append 모드
    train_steps = len(train_loader)

    model.train()
    if args.ML == 'ML':
        ML_model = xgb.XGBRegressor(objective='reg:squarederror')
        ML_model = LGBMRegressor()
        
    progress = tqdm(range(args.epoch))
    for i in progress:
        
        if args.ML == 'ML':
            train_losses, ML_m = train(model, train_loader, i,ML = args.ML, ML_model = ML_model)
        else:
            train_losses, _ = train(model, train_loader, i,ML = args.ML)
            
        # Validation loop
        val_loss = validation(model, val_loader,ML = args.ML)
        val_losses.append(val_loss)
        
        if i % 10 == 0:
            torch.save(model.state_dict(), f'{save_dir}/model_{time_str}_ep{i}_loss{val_loss}.pth')
            
        
    # 텐서를 numpy 배열로 변환할 때, CUDA 텐서를 CPU로 이동 후 변환
    
    train_losses_cpu = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    val_losses_cpu = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]
    torch.save(model.state_dict(), f'{save_dir}/model_{time_str}.pth')
    log_file.close()
    if args.ML == 'ML':
        ML_m.save_model(f'{save_dir}/ML_model_{time_str}.json')
    
    # save results
    np.save(os.path.join(save_dir, f'train_losses_{time_str}.npy'), np.array(train_losses_cpu))
    np.save(os.path.join(save_dir, f'val_losses_{time_str}.npy'), np.array(val_losses_cpu))
        

if args.mode == 'test':
    # Test loop
    model_path = f'{save_dir}/{args.test_dir}.pth'
    #model_path = f'{save_dir}/model_2024-09-23_17-03-16.pth'
    ml_path = f'{save_dir}/ML_model_2024-09-20_15-03-04.json'
    
        
    test_loss, loss_per_subject,plot = test(model, test_loader, model_path,ml_path,ML = args.ML)
    test_loss_cpu = test_loss.cpu().item() if isinstance(test_loss, torch.Tensor) else test_loss
    loss_per_subject_cpu = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in loss_per_subject]

    # Save test results
    np.save(os.path.join(save_dir, f'test_loss_{time_str}.npy'), np.array(test_loss_cpu))
    np.save(os.path.join(save_dir, f'loss_per_subject_{time_str}.npy'), np.array(loss_per_subject_cpu))
    np.save(os.path.join(save_dir, f'plot_hcp.npy'), plot)

if args.mode == 'train_future':
    model.train()
        
    progress = tqdm(range(args.epoch))
    for i in progress:
        
        if args.ML == 'ML':
            train_losses, ML_m = train_future(model, train_future_loader, i,ML = args.ML, ML_model = ML_model)
        else:
            train_losses, _ = train_future(model, train_future_loader, i,ML = args.ML)
            
        # Validation loop
        val_loss = validation_future(model, val_future_loader,ML = args.ML)
        val_losses.append(val_loss)
        
        if i % 10 == 0:
            torch.save(model.state_dict(), f'{save_dir}/model_{time_str}_ep{i}_loss{val_loss}.pth')
            
        
    # 텐서를 numpy 배열로 변환할 때, CUDA 텐서를 CPU로 이동 후 변환
    
    train_losses_cpu = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    val_losses_cpu = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]
    torch.save(model.state_dict(), f'{save_dir}/model_{time_str}.pth')
    if args.ML == 'ML':
        ML_m.save_model(f'{save_dir}/ML_model_{time_str}.json')
    
    # save results
    np.save(os.path.join(save_dir, f'train_losses_{time_str}.npy'), np.array(train_losses_cpu))
    np.save(os.path.join(save_dir, f'val_losses_{time_str}.npy'), np.array(val_losses_cpu))

    
    
    
if args.mode == 'test_future':
    # Test loop
    model_path = f'{save_dir}/{args.test_dir}.pth'
    #model_path = f'{save_dir}/model_2024-09-23_17-03-16.pth'
    ml_path = f'{save_dir}/ML_model_2024-09-20_15-03-04.json'
    save_dir_future =  f'{save_dir}'
        
    test_loss, loss_per_subject,plot = test_future(model, test_future_loader, model_path,ml_path)
    test_loss_cpu = test_loss.cpu().item() if isinstance(test_loss, torch.Tensor) else test_loss
    loss_per_subject_cpu = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in loss_per_subject]

    # Save test results
    np.save(os.path.join(save_dir_future, f'test_loss_{time_str}.npy'), np.array(test_loss_cpu))
    np.save(os.path.join(save_dir_future, f'loss_per_subject_{time_str}.npy'), np.array(loss_per_subject_cpu))
    np.save(os.path.join(save_dir_future, f'plot_{time_str}.npy'), plot)








        