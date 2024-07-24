esc50_path = "data/esc50"
# esc50_path = 'D:/sound_datasets/esc50'
runs_path = "results"
# sub-epoch (batch-level) progress bar display
disable_bat_pbar = False  # True

# do not change this block
n_classes = 50
folds = 5
test_folds = [1, 2, 3, 4, 5]
# ratio to split off from training data
#val_size = 0.1  # could be changed

n_channels = 3


# model_constructor = "ResNet(block=ResidualBlock, layers=[2, 2, 2, 2], num_classes=config.n_classes)"
model_constructor = "GetACDNetModel()"

# model checkpoints loaded for testing
test_checkpoints = ["terminal.pt"]#, "best_val_acc.pt"]
# experiment folder used for testing (result from cross validation training)
test_experiment = "results/ACDNet-Run2-Laptop"
# test_experiment = "results/sample-run"

# Param1
sr = 20000
inputLength = 30225
nCrops = 10

device_id = 0
batch_size = 64
num_workers = 0  # 16
persistent_workers = False
epochs = 2000
patience = 2000
lr = 0.1
weight_decay = 5e-4
warm_epochs = 10
gamma = 0.9
step_size = 10
momentum = 0.9
schedule = [0.3, 0.6, 0.9]