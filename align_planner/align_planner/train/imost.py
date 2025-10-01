import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from align_planner.train.trainutils import DistanceMetrics 


class ImostDataBuffer:
    def __init__(self, KNN_hyperparameter, lambda_param_for_js_filtering, device='cuda'):
        """
        Initialize the buffer with the specified parameters.
        
        Args:
        KNN_hyperparameter: The number of neighbors for KNN.
        lambda_param_for_js_filtering: Threshold for Jensen-Shannon filtering.
        device: Device to run the computations ('cpu' or 'cuda').
        """
        self.device = device  # Set the device (CPU or GPU)
        self.input_clusters = []  # stores clusters of input-output pairs
        self.KNN_hyperparameter = KNN_hyperparameter
        self.lambda_param_for_js_filtering = lambda_param_for_js_filtering
        self.kneighbors = NearestNeighbors(n_neighbors=self.KNN_hyperparameter)
        self.dist_measure =  DistanceMetrics()


    def store_dataloader(self, dataloader: DataLoader):
        for batch_idx, (new_input_batch, new_output_batch) in enumerate(dataloader):            
            if new_input_batch.shape[0] == 1:
                continue
            new_input_batch = new_input_batch.to(self.device)
            new_output_batch = new_output_batch.to(self.device)
            self.append_batch(new_input_batch, new_output_batch)            
       
    def kl_divergence(self, mu_p, std_p, mu_q, std_q):
        """
        Compute Kullback-Leibler Divergence between two normal distributions.
        """
        return torch.log(std_q / std_p) + (std_p ** 2 + (mu_p - mu_q) ** 2) / (2 * std_q ** 2) - 0.5

    def jensen_shannon_divergence(self,mu_p, std_p, mu_q, std_q):
        """
        Compute Jensen-Shannon Divergence between two normal distributions using their means and stds.
        """
        # Compute the average distribution
        mu_m = (mu_p + mu_q) / 2
        std_m = (std_p + std_q) / 2

        # Compute KL divergence for P || M and Q || M
        kl_pm = self.kl_divergence(mu_p, std_p, mu_m, std_m)
        kl_qm = self.kl_divergence(mu_q, std_q, mu_m, std_m)

        # Jensen-Shannon Divergence (symmetric)
        jsd = 0.5 * (kl_pm + kl_qm)
        
        return jsd


    def batch_to_cluster(self, new_input_batch, new_output_batch):
        """
        Create a cluster based on the new input-output batch.
        A cluster contains the new inputs, outputs, and the mean of the new input batch.
        """
        mean = new_input_batch.mean(dim=0)
        std = new_input_batch.std(dim=0)
        cluster = {
            "input": new_input_batch,
            "output": new_output_batch,
            "mean": mean.to(self.device),  # Ensure mean is on the correct device
            "std": std.to(self.device)  # Ensure std is on the correct device
        }
        return cluster

    def compute_jensen_shannon(self, new_cluster):
        """
        Optimized computation of Jensen-Shannon divergence for each stored cluster.
        """
        distances = []
        new_mean = new_cluster["mean"]
        new_std = new_cluster["std"]
        new_var = 2 * torch.log(new_std)  # Log variance

        for cluster in self.input_clusters:
            stored_mean = cluster["mean"]
            stored_std = cluster["std"]            
            stored_var = 2 * torch.log(stored_std)  # Log variance            
            distance = self.dist_measure.jensen_shannon_divergence_torch(new_mean, new_var, stored_mean, stored_var)             
            distances.append(distance)
        
        return distances

    def get_clusters(self, new_input_batch):
        """
        Apply KNN clustering to the new input batch and return the closest clusters.
        """
        if len(self.input_clusters) > 0:
            cluster_means = torch.stack([cluster["mean"] for cluster in self.input_clusters]).cpu().numpy()
            self.kneighbors.fit(cluster_means)  
            distances, indices = self.kneighbors.kneighbors(new_input_batch.cpu().numpy())
            clusters = [self.input_clusters[i] for i in indices.flatten()]
            return clusters
        else:
            return []

    def init_data_clustering(self, new_input_batch, new_output_batch):
        """
        Initialize data clustering by processing the new input-output batch and storing the resulting clusters.
        """
        new_cluster = self.batch_to_cluster(new_input_batch, new_output_batch)
        self.input_clusters.append(new_cluster)

    def append_batch(self, new_input_batch, new_output_batch):
        """
        Append a new batch to the buffer if it is sufficiently distinct (based on JS divergence).
        """
        if len(self.input_clusters) == 0:
            self.init_data_clustering(new_input_batch, new_output_batch)
            return 
        new_cluster = self.batch_to_cluster(new_input_batch, new_output_batch)
        distances_to_collected_clusters = self.compute_jensen_shannon(new_cluster)

        if min(distances_to_collected_clusters) > self.lambda_param_for_js_filtering:
            self.input_clusters.append(new_cluster)

    def recall_cluster_data(self):
        """
        Randomly return an input-output pair from the stored clusters.
        """        
        random_index = torch.randint(0, len(self.input_clusters), (1,)).item()    
        cluster = self.input_clusters[random_index]
        return cluster["input"], cluster["output"]


