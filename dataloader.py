import numpy as np
import torch
from torch.utils.data import  Dataset
from argument import args_parser

args = args_parser()
def hcp_data_load(atlas, data_dir):
    
    dataset_1 = np.load(data_dir + '/HCP_s'+atlas+'_data_seg0.npy', allow_pickle=True)
    dataset_2 = np.load(data_dir + '/HCP_s'+atlas+'_data_seg1.npy', allow_pickle=True)

    dataset = np.concatenate((dataset_1, dataset_2), axis=0)   
    
    
    #dataset = np.load('/camin1/yrjang/HCP_data/HCP/retest_atlas/Schaefer300_4runs.npy', allow_pickle=True) 
    #dataset = np.load('/camin1/yrjang/HCP_data/eNKI/eNKI_dataset.npy', allow_pickle=True) 
    data_x = []
    
    for data in dataset:
        if args.other_data =='HCP':
            if data['roiTimeseries'].shape[0] != 4800:
                continue
        #if data['roiTimeseries'].shape[0] != 884:
        #    print('not 884')
        #    continue
        data_x.append(data['roiTimeseries'])

    data_x = np.array(data_x)
    print("Data shape:", data_x.shape)
    return data_x
    
    
    
def data_split(data,train_size, val_size, test_size, random_state=3):
    # data shuffle
    if random_state is not None:
        np.random.seed(random_state)
    
    np.random.shuffle(data)
    
    n_total = len(data)
    
    # split data
    n_train = int(train_size * n_total)
    n_val = int(val_size * n_total)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    print("Train data shape:", train_data.shape)
    print("Validation data shape:", val_data.shape)
    print("Test data shape:", test_data.shape)

    #np.save('train_HCP_data.npy', train_data)
    #print("Train data saved as train_data.npy")

    #np.save('test_HCP_data.npy', test_data)
    print("Test data saved as test_data.npy")
    
    return train_data, val_data, test_data
    
    

class WindowDataset(Dataset):
    def __init__(self, data, input_window, output_window, stride, batch_size):
        """
        Args:
            data: Time series data (shape: (num_subjects, total_timepoints, num_features)).
                  If the data is too large to fit in memory, you can use np.load(..., mmap_mode='r') to enable memory mapping.
            input_window (int): Length of the input window.
            output_window (int): Length of the output window.
            stride (int): Sliding interval between windows.
        """
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.stride = stride
        
        self.index_list = []  # (subject_index, start_idx) pairs to store
        
        n_sub = data.shape[0]         # # of subjects
        ts_length = data.shape[1]     # time series length per subject
        roi = data.shape[2]          # feature dimension (number of ROIs)

        
        for subj_idx in range(n_sub):
            max_start = ts_length - (input_window + output_window)
            if max_start < 0:
                print(f"Subject {subj_idx} has too short time series (length: {ts_length})")
                continue
            num_samples = (max_start // stride) + 1
            
            for i in range(num_samples):
                start_x = stride * i
                self.index_list.append((subj_idx, start_x))
        print(f"Total {len(self.index_list)} samples in the dataset")

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        """
        After extracting (subject_idx, start_idx) from index_list,
        generate x and y on-the-fly by slicing only that segment.
        """
        subj_idx, start_x = self.index_list[idx]
        end_x = start_x + self.input_window
        
        # x window
        x = self.data[subj_idx, start_x:end_x, :]
        
        # y window
        start_y = end_x
        end_y = start_y + self.output_window
        y = self.data[subj_idx, start_y:end_y, :]

        # numpy -> torch tensor 변환
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).float()
        #print('x_tensor:', x_tensor.shape)
        #print('y_tensor:', y_tensor.shape)

        return x_tensor, y_tensor
    
    

    
    

class SortedBatchWindowDataset(Dataset):
    def __init__(self, data, input_window, output_window, stride, batch_size):
        """
        Args:
            data: Time series data (shape: (num_subjects, total_timepoints, num_features)).
                  If the data is too large to fit in memory, you can use np.load(..., mmap_mode='r') to enable memory mapping.
            input_window (int): Length of the input window.
            output_window (int): Length of the output window.
            stride (int): Interval for sliding the window.
            batch_size (int): Used to adjust the number of extracted samples per subject to be a multiple of the batch size.
        """
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.stride = stride
        self.batch_size = batch_size
        
        self.index_list = []  
        
        n_sub = data.shape[0]     
        ts_length = data.shape[1]  
        
        for subj_idx in range(n_sub):
            
            max_start = ts_length - (input_window + output_window)
            if max_start < 0:
                print(f"Subject {subj_idx} has too short time series (length: {ts_length}). Skipping.")
                continue
            num_samples = (max_start // stride) + 1
            print(f"Subject {subj_idx}: ts_length={ts_length}, num_samples={num_samples}")
            
            for i in range(num_samples):
                start_x = stride * i
                self.index_list.append((subj_idx, start_x))
        print(f"Total {len(self.index_list)} samples in the dataset")

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        """
        After retrieving (subject_idx, start_idx) from index_list,
        slice the corresponding segment to generate x and y windows on-the-fly,
        and return the associated subject ID as well.
        """

        subj_idx, start_x = self.index_list[idx]
        end_x = start_x + self.input_window
        x = self.data[subj_idx, start_x:end_x, :]

        start_y = end_x
        end_y = start_y + self.output_window
        y = self.data[subj_idx, start_y:end_y, :]

        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).float()

        return x_tensor, y_tensor, subj_idx
