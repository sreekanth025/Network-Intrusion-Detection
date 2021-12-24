
class Args:
    def __init__(self):
        self.epochs = 100
        self.lr = 0.01
        self.n_columns = 33
        self.agent_data_splits = 50
        self.random_state = 42
        self.test_set_size = 0.1
        self.batch_size=64
        
args = Args()

# Exports: args