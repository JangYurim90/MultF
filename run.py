import torch
import torch.nn as nn
import psutil
import time
import os
from argument import args_parser




args = args_parser()
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


def train(model, train_loader,optimizer,criterion,scheduler,device,progress,i):
    batchloss = 0.0
    print("Epoch: ", i)
    model.to(device)
    model = nn.DataParallel(model, device_ids=[0,1])
    iter_count = 0
    time_now = time.time()
    train_losses = []


    for i_e, (inputs, outputs) in enumerate(train_loader):
        iter_count += 1
        optimizer.zero_grad(set_to_none=True)
        
        # Inputs to encoder (src), and targets for decoder (tgt)
        src = inputs.float().to(device)  # Input timepoints (e.g., 100 timepoints)
        tgt = outputs.float().to(device)
        

        src_mask = generate_padding_mask_bool(batch_size=src.shape[0],seq_length=src.shape[1]).to(device)
        tgt_mask = generate_padding_mask(batch_size=args.batch_size,seq_length=tgt.shape[0]).to(device)
        
        
        # Get the output and Q, K, V from transformer encoder
        result, all_q, all_k,all_v = model(src, src_mask)
        
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

    
    if len(train_loader) > 0:
        progress.set_description("{:0.5f}".format(batchloss / len(train_loader)))
        print(f"lr: {scheduler.get_last_lr()[0]}")
        train_losses.append(batchloss / len(train_loader))
    else:
        progress.set_description("No data in train_loader")
        
    return train_losses

def validation(model, val_loader,optimizer,criterion,scheduler,device):
    model.to(device)
    model = nn.DataParallel(model, device_ids=[0,1])
    
    model.eval()
    total_val_loss = 0.0
    

    with torch.no_grad():
        for (inputs, outputs) in val_loader:
        
            src = inputs.float().to(device)
            tgt = outputs.float().to(device)

            src_mask = generate_padding_mask_bool(batch_size=src.shape[0],seq_length=src.shape[1]).to(device)
            tgt_mask = generate_padding_mask(batch_size=args.batch_size,seq_length=tgt.shape[0]).to(device)
            
            # Get the output and Q, K, V from both encoder and decoder
            result, all_q, all_k,all_v = model(src, src_mask)
            outputs = outputs.to(device)

            val_loss = criterion(result, outputs.float().to(device))
            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(val_loader)
    print(f"Validation Loss: {average_val_loss}")
    
    return average_val_loss
    
def test(model, test_loader,model_path,optimizer,criterion,scheduler,device):
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
            
            
            # Get the output and Q, K, V from both encoder and decoder
            result, all_q, all_k,all_v = model(src, src_mask)

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