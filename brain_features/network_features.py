from bct import participation_coef, community_louvain
import numpy as np

class Network_Features:
    def __init__(self, connectivity_matrix ,gradients = None, module_paths=None, seed=4):
        """
        Initializes the Network_Features class with atlas values and connectivity matrix.
        
        Args:
            atlas_values (np.ndarray): Atlas values indicating network.
            connectivity_matrix (np.ndarray): Connectivity matrix of the brain network.
            gradients (np.ndarray): Gradients of connectivity matrix for the brain regions.
            module_paths (str): Path to the module data for louvain algorithm. If None, louvain will not be applied.
            seed (int): Random seed for reproducibility.
        Returns:
            None
        """
        self.connectivity_matrix = connectivity_matrix
        self.gradients = gradients
        self.module_paths = module_paths
        self.seed = seed

        # louvain algorithm
        if module_paths is not None:
            module_data = np.load(module_paths, allow_pickle=True)
            [m, Q] = community_louvain(np.mean(module_data, axis=0), self.seed)
            self.modules = m
            self.Q = Q
            print("NumMod: ", len(np.unique(m)))
            print('Modularity: {0:.4f}'.format(Q))  



    ## Network features 
    # Network hierarchy /louvain algorithm / participation coefficient
    
    def Network_hierarchy(self,atlas_values, gradients= None):
        """
        Calculates the network hierarchy based on the atlas values and connectivity matrix.
        Args:
            atlas_values (np.ndarray): Atlas values indicating network.
            gradients (np.ndarray): Gradients of connectivity matrix for the brain regions.
        Returns:
            hierarchy (np.ndarray): Network hierarchy for each node.
        """
        label_txt = ['Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention',
                'Limbic', 'Frontoparietal', 'Default']
        Visual = np.where(atlas_values == 1)[0]
        Somatomotor = np.where(atlas_values == 2)[0]
        Limbic = np.where(atlas_values == 5)[0]
        Frontoparietal = np.where(atlas_values == 6)[0]
        Default = np.where(atlas_values == 7)[0]

        low_level = np.concatenate((Visual, Somatomotor))
        high_level2 = np.concatenate((Limbic, Frontoparietal, Default))

        network2 = [low_level, high_level2]

        if gradients is None:
            gradients = self.gradients

        if gradients.ndim == 3:
            hierarchy_value = []
            for i in range(gradients.shape[0]):
                network_ = np.array(gradients[i,:,0])
                sensory_network = np.mean(network_[low_level])
                high_network = np.mean(network_[high_level2])
                hierarchy_value.append(sensory_network - high_network)
            hierarchy = np.array(hierarchy_value)
        elif gradients.ndim == 2:
            network_ = np.array(gradients[:,0])
            sensory_network = np.mean(network_[low_level])
            high_network = np.mean(network_[high_level2])
            hierarchy = sensory_network - high_network

        else:
            raise ValueError("Gradients must be either 2D or 3D array.")
        
        return hierarchy
    

    def participation_coefficient(self, connectivity_matrix = None, modules=None):
        """
        Calculates the participation coefficient of the network.
        
        Args:
            connectivity_matrix (np.ndarray): Connectivity matrix of the brain network.
            modules (np.ndarray): Module assignments for each node.
        
        Returns:
            participation_coef (np.ndarray): Participation coefficient for each node.
        """
        if connectivity_matrix is None:
            connectivity_matrix = self.connectivity_matrix

        if modules is None:
            modules = self.modules

        if connectivity_matrix.ndim == 3:
            pc = []
            for i in range(connectivity_matrix.shape[0]):
                pc_i = participation_coef(connectivity_matrix[i], modules)
                pc.append(pc_i)
            participation_coef = np.array(pc)
        elif connectivity_matrix.ndim == 2:
            # If the input is a 2D matrix, calculate participation coefficient directly
            participation_coef = participation_coef(connectivity_matrix, modules)

        else:
            raise ValueError("Connectivity matrix must be either 2D or 3D array.")

        return participation_coef
