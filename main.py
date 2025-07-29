#!/usr/bin/env python3

import numpy as np
import os
from torch.utils.data import DataLoader
from argument import args_parser
from dataloader import data_split,hcp_data_load, WindowDataset, SortedBatchWindowDataset
import torch
from torch import nn
from models_encoder.layers import TSTransformerEncoder
from tqdm import tqdm
from datetime import datetime
import time
import psutil
from run import train, validation, test



# if the directory does not exist, create it
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_code_as_txt(source_file, destination_path):
    try:
        with open(source_file, 'r') as source:
            code = source.read()
        
        with open(destination_path, 'w') as destination:
            destination.write(code)
        
        print(f"Code has been successfully saved to {destination_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")


def get_time_string():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        
def lr_lambda(current_step):
    # warmup step: current step < warmup_steps -> linear increase
    if current_step < warmup_steps:
        return float(current_step) / float(max(1,warmup_steps))
    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))



args = args_parser()
device = torch.device("cuda:0")

# Save the model with the lowest validation loss
save_dir = "{}/A{}_i{}_o{}_lr{}_nl{}_sub{}_ep{}_{}_MSE_tanh".format(args.loss_dir, args.atlas, args.i_win, args.o_win, args.lr,args.nlayers,80,args.epoch)
create_directory_if_not_exists(save_dir)

source_file = '/camin1/yrjang/Brain_network_dynamics/Dynamics_RNN/argument.py'
destination_path = f'{save_dir}/hyperparameter.txt'
save_code_as_txt(source_file, destination_path)

time_str = get_time_string()  

load_data = hcp_data_load(args.atlas,args.HCPdata_dir)

if args.other_data != 'other_data':
    # data split
    train_data, val_data, test_data = data_split(load_data,args.train_size,args.val_size,args.test_size,random_state = 3)
else:
    test_data = load_data


# data_loader 
if args.mode == 'train':
    del test_data
    
    print('Start loader')
    
    train_dataset = WindowDataset(train_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride, batch_size = args.batch_size)
    val_dataset = WindowDataset(val_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride,batch_size = args.batch_size)

    del train_data
    del val_data
    
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,drop_last=True)
    
    del train_dataset
    del val_dataset

    print('End loader')

elif args.mode == 'test':
    
    if args.other_data != 'other_data':
        del train_data
        del val_data
    
    test_dataset = SortedBatchWindowDataset(test_data,input_window = args.i_win,output_window = args.o_win,stride = args.stride , batch_size = args.batch_size)

    del test_data
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,drop_last=False)
    del test_dataset
    



# Load model (transformer)
model = TSTransformerEncoder(feat_dim=args.feature_dim, max_len =args.i_win,d_model=args.d_model,n_heads=args.nhead,num_layers=args.nlayers,dim_feedforward=args.feature_dim ,linear_hidden=args.linear_hidden,output_length=args.o_win,dropout=args.dropout).to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
training_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(
    f"Total parameters: {total_params:,}, "
    f"Trainable parameters: {training_params:,}"
)


criterion = nn.MSELoss(reduction='mean')

if args.mode == 'train':
    total_steps = len(train_loader) * args.epoch
    warmup_steps = len(train_loader) * args.warmup_steps

elif args.mode == 'test':
    total_steps = len(test_loader) * args.epoch
    warmup_steps = len(test_loader) * args.warmup_steps
    

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
    progress = tqdm(range(args.epoch))

    for i in progress:
        train_losses = train(model, train_loader, optimizer,criterion,scheduler,device,progress,i)
            
        # Validation loop
        val_loss = validation(model, val_loader,optimizer,criterion,scheduler,device)
        val_losses.append(val_loss)
        
        if i % 10 == 0:
            torch.save(model.state_dict(), f'{save_dir}/model_{time_str}_ep{i}_loss{val_loss}.pth')
            
        
    # 텐서를 numpy 배열로 변환할 때, CUDA 텐서를 CPU로 이동 후 변환
    
    train_losses_cpu = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    val_losses_cpu = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]
    torch.save(model.state_dict(), f'{save_dir}/model_{time_str}.pth')
    log_file.close()

    
    # save results
    np.save(os.path.join(save_dir, f'train_losses_{time_str}.npy'), np.array(train_losses_cpu))
    np.save(os.path.join(save_dir, f'val_losses_{time_str}.npy'), np.array(val_losses_cpu))
        

if args.mode == 'test':
    # Test loop
    model_path = f'{save_dir}/{args.test_dir}.pth'
    
    test_loss, loss_per_subject,plot = test(model, test_loader, model_path,optimizer,criterion,scheduler,device)
    test_loss_cpu = test_loss.cpu().item() if isinstance(test_loss, torch.Tensor) else test_loss
    loss_per_subject_cpu = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in loss_per_subject]

    # Save test results
    np.save(os.path.join(save_dir, f'test_loss_{time_str}.npy'), np.array(test_loss_cpu))
    np.save(os.path.join(save_dir, f'loss_per_subject_{time_str}.npy'), np.array(loss_per_subject_cpu))
    np.save(os.path.join(save_dir, f'plot_hcp.npy'), plot)










        