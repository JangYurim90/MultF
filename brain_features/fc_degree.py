import numpy as np
from nilearn.connectome import ConnectivityMeasure




def organize_output(path, timepoints,input_window, output_window, stride, n_sub, roi):
    test_plot = np.load(path, allow_pickle=True)

    window_size = input_window + output_window
    num_windows = 0 
    start = 0

    while start + window_size <= timepoints:
        num_windows += 1
        start += stride
    print("Total number of windows:", num_windows)

    test_plot_results = test_plot.item()['result']
    test_plot_output = test_plot.item()['output']
    cutting = len(test_plot_results)-1

    # predict array
    test_plot_results_f = np.array(test_plot_results[:cutting])
    test_plot_results_l = np.array(test_plot_results[cutting])

    test_plot_results_f = test_plot_results_f.reshape(-1,output_window,roi)
    test_plot_results_combine = np.concatenate((test_plot_results_f,test_plot_results_l),axis=0)

    # actual array
    test_plot_output_f = np.array(test_plot_output[:cutting])
    test_plot_output_l = np.array(test_plot_output[cutting])

    test_plot_output_f = test_plot_output_f.reshape(-1,output_window,roi)
    test_plot_output_combine = np.concatenate((test_plot_output_f,test_plot_output_l),axis=0)

    print("Results shape:", test_plot_results_combine.shape)
    print("Output shape:", test_plot_output_combine.shape)

    sub_predict = np.array(test_plot_results_combine).reshape(n_sub,-1,roi)
    sub_actual = np.array(test_plot_output_combine).reshape(n_sub,-1,roi)

    print("sub_predict shape: ", sub_predict.shape)
    print("sub_actual shape: ", sub_actual.shape)

    return sub_predict, sub_actual



def thresholding(matrix):
    # thresholding 90% throw away
    rows, cols = matrix.shape
    result = np.zeros_like(matrix)  
    
    for i in range(rows):
        row = matrix[i, :]
        threshold = np.percentile(row, 90)  
        result[i, row >= threshold] = row[row >= threshold]
    return result


def cal_FC(data):
    # calculate FC
    
    np.seterr(divide='ignore', invalid='ignore')

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([data])[0]

    # z-transform
    correlation_matrix = np.arctanh(correlation_matrix)
    np.fill_diagonal(correlation_matrix,0)

    # thresholding
    correlation_matrix = thresholding(correlation_matrix)
    return correlation_matrix

def cal_FC_z(data):
    # calculate FC with z-transform without thresholding
    np.seterr(divide='ignore', invalid='ignore')
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([data])[0]
    
    # z-transform
    correlation_matrix = np.arctanh(correlation_matrix)
    np.fill_diagonal(correlation_matrix,0)
    
    return correlation_matrix

def degree_centrality(matrix):
    # calculate degree centrality
    degree = np.sum(matrix, axis=1)
    return degree