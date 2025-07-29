import numpy as np
from nilearn.connectome import ConnectivityMeasure
from brainspace.gradient import GradientMaps


class Brain_features:
    """
    This class is designed to handle brain feature extraction from time series data.
    It organizes the output data into a structured format suitable for further analysis.
    """
    def __init__(self, path, timepoints, input_window, output_window, stride, n_sub, roi):
        """
        Initializes the brain_features class with parameters for organizing output data.

        Args:
            path (str): Path to the data file.
            timepoints (int): Total number of time points in the data.
            input_window (int): Length of the input window.
            output_window (int): Length of the output window.
            stride (int): Stride length for sliding window.
            n_sub (int): Number of subjects.
            roi (int): Number of regions of interest.
        """
        self.timepoints = timepoints
        self.input_window = input_window
        self.output_window = output_window
        self.stride = stride
        self.n_sub = n_sub
        self.roi = roi

        self.test_plot = np.load(path, allow_pickle=True)
        self.window_size = input_window + output_window
        self.num_windows = 0
        self.start = 0
        while self.start + self.window_size <= timepoints:
            self.num_windows += 1
            self.start += stride
        print("Total number of windows:", self.num_windows)

        self.test_plot_results = self.test_plot.item()['result']
        self.test_plot_output = self.test_plot.item()['output']
        cutting = len(self.test_plot_results) - 1

        # Predict array
        test_plot_results_f = np.array(self.test_plot_results[:cutting])
        test_plot_results_l = np.array(self.test_plot_results[cutting])
        test_plot_results_f = test_plot_results_f.reshape(-1, self.output_window, self.roi)
        test_plot_results_combine = np.concatenate((test_plot_results_f, test_plot_results_l), axis=0)

        # Actual array
        test_plot_output_f = np.array(self.test_plot_output[:cutting])
        test_plot_output_l = np.array(self.test_plot_output[cutting])
        test_plot_output_f = test_plot_output_f.reshape(-1, self.output_window, self.roi)
        test_plot_output_combine = np.concatenate((test_plot_output_f, test_plot_output_l), axis=0)

        self.sub_predict = np.array(test_plot_results_combine).reshape(self.n_sub, -1, self.roi)
        self.sub_actual = np.array(test_plot_output_combine).reshape(self.n_sub, -1, self.roi)

        print("sub_predict shape: ", self.sub_predict.shape)
        print("sub_actual shape: ", self.sub_actual.shape)


    def __getitem__(self, idx):
        """
        Retrieves the prediction and actual output for a given subject index.
        Just timeseries data is returned, not the features

        Args:
            idx (int): Index of the subject.

        Returns:
            tuple: Contains the predicted and actual outputs for the specified subject.
        """
        return self.sub_predict[idx], self.sub_actual[idx]

    
    def thresholding(self, matrix):
        """
        Applies thresholding to the input matrix, keeping only the top 10% of values.

        Args:
            matrix (np.ndarray): Input matrix to be thresholded.

        Returns:
            result (np.ndarray): Thresholded matrix.
        """
        rows, cols = matrix.shape
        result = np.zeros_like(matrix)

        for i in range(rows):
            row = matrix[i, :]
            threshold = np.percentile(row, 90)
            result[i, row >= threshold] = row[row >= threshold]
        return result
    
    def cal_FC(self, data = None, type = 'predict'):
        """
        Calculates the functional connectivity (FC) matrix for the given data.
        Args:
            data (np.ndarray): Input time series data. If None, uses the stored predictions.
            type (str): Type of data to process ('predict' or 'actual').
            Returns:
                correlation_matrix (np.ndarray): Functional connectivity matrix after z-transform and thresholding.
        """
        np.seterr(divide='ignore', invalid='ignore')

        correlation_measure = ConnectivityMeasure(kind='correlation')

        if data is None:
            correlation_matrix = np.zeros((self.n_sub, self.roi, self.roi))
            if type == 'predict':
                for i in range(self.n_sub):
                    correlation_matrix = correlation_measure.fit_transform([self.sub_predict[i]])[0]

                    correlation_matrix = np.arctanh(correlation_matrix)
                    np.fill_diagonal(correlation_matrix, 0)

                    correlation_matrix = self.thresholding(correlation_matrix)
            else: # actual
                for i in range(self.n_sub):
                    correlation_matrix = correlation_measure.fit_transform([self.sub_actual[i]])[0]

                    correlation_matrix = np.arctanh(correlation_matrix)
                    np.fill_diagonal(correlation_matrix, 0)

                    correlation_matrix = self.thresholding(correlation_matrix)
        else:
            # If data is provided, calculate FC for the given data
            # Only one subject can be processed at a time
            correlation_matrix = correlation_measure.fit_transform([data])[0]

            # z-transform
            correlation_matrix = np.arctanh(correlation_matrix)
            np.fill_diagonal(correlation_matrix, 0)

            # thresholding
            correlation_matrix = self.thresholding(correlation_matrix)

        return correlation_matrix
    

    def cal_FC_z(self, data=None, type='predict'):       
        """
        Calculates the functional connectivity (FC) matrix with z-transform without thresholding.
        Args:
            data (np.ndarray): Input time series data. If None, uses the stored predictions.
            type (str): Type of data to process ('predict' or 'actual').
        Returns:
            correlation_matrix (np.ndarray): Functional connectivity matrix after z-transform.
        """
        np.seterr(divide='ignore', invalid='ignore')
        correlation_measure = ConnectivityMeasure(kind='correlation')

        if data is None:
            correlation_matrix = np.zeros((self.n_sub, self.roi, self.roi))
            if type == 'predict':
                for i in range(self.n_sub):
                    correlation_matrix = correlation_measure.fit_transform([self.sub_predict[i]])[0]
                    
                    # z-transform
                    correlation_matrix = np.arctanh(correlation_matrix)
                    np.fill_diagonal(correlation_matrix, 0)
            else:  # actual
                for i in range(self.n_sub):
                    correlation_matrix = correlation_measure.fit_transform([self.sub_actual[i]])[0]
                    
                    # z-transform
                    correlation_matrix = np.arctanh(correlation_matrix)
                    np.fill_diagonal(correlation_matrix, 0)

        else:
            # If data is provided, calculate FC for the given data
            # Only one subject can be processed at a time
            correlation_matrix = correlation_measure.fit_transform([data])[0]
            
            # z-transform
            correlation_matrix = np.arctanh(correlation_matrix)
            np.fill_diagonal(correlation_matrix, 0)
        
        return correlation_matrix
    

    def degree_centrality(self, matrix):
        """
        Calculates the degree centrality of the input matrix.

        Args:
            matrix (np.ndarray): Input connectivity matrix.

        Returns:
            degree (np.ndarray): Degree centrality for each node.
        """
        # calculate degree centrality

        if matrix.ndim == 3:
            degree = np.sum(matrix, axis=2)
            
        elif matrix.ndim == 2:
            # If the input is a 2D matrix, calculate degree centrality directly
            degree = np.sum(matrix, axis=1)

        else:
            raise ValueError("Input matrix must be either 2D or 3D array.")
        
        return degree
        

    def compute_gradient(data, reference_file,kernal='normalized_angle', approach='dm', alignment='procrustes', n_components=3 ):
        align_= GradientMaps(kernel=kernal, approach=approach, alignment=alignment, n_components=n_components)

        if data.ndim == 3:
            gradient_ = np.zeros((data.shape[0], data.shape[1], n_components))
            for i in range(data.shape[0]): # Loop through subjects
                align_.fit(data[i], reference = reference_file)
                gradient_[i] = align_.aligned_

        elif data.ndim == 2:
            # If data is 2D, fit the gradient directly
            align_.fit(data, reference = reference_file)
            gradient_ = align_.aligned_

        else:
            raise ValueError("Data must be either 2D or 3D array.")
        return gradient_
    
    
    
