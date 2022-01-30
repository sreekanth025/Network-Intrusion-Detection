
class Args:
    def __init__(self):
        self.epochs = 100
        self.lr = 0.01
        self.agent_data_splits = 20
        self.random_state = 42
        self.test_set_size = 0.1
        self.batch_size = 64
        
        # Hyper parameter in prioritized experience replay
        self.per_exponent = 2
        
        # self.num_clients = 3
        # self.num_clients = 5
        self.num_clients = 5
        
        self.output_folder = 'script_outputs/'
        self.file_suffix = '-output.txt'
        
# =============================================================================
#         for nsl dataset:
# =============================================================================
        # self.n_columns = 33
        
        # # self.fparam_k = 4500
        # # self.fparam_a = 500
        
        # self.fparam_k = 10
        # self.fparam_a = 10
        
# =============================================================================
#         for isot dataset
# =============================================================================
        # self.n_columns = 208
        self.n_columns = 79
        
        self.fparam_k = 10
        self.fparam_a = 10
        
        
        # self.prev_multiplier_weight = 0.5
        # self.re_epochs = 25
        self.multiplier_factor = 1
        
args = Args()

# Exports: args