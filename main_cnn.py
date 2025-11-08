# Last Update 2025 09 06 
import torch
from torch.utils.data import DataLoader,Subset
from torch.optim import Adam
import os
import json
# import nni 

from opnn_cnn import opnn


from utils import log_loss, save_loss_to_dated_file, plot_logs,plot_prediction,store_model
from utils import ensure_directory_exists,get_time_YYYYMMDDHH
from trainer import Trainer, load_data_by_split
import argparse
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

EXPECTED_IMG_SIZE = (162, 512)
EXPECTED_SIM_SIZE = (162, 512)


with open('config.json', 'r') as file:
    config = json.load(file)
DATA_PATH = config['DATA_PATH']
RESULT_FOLDER = config['RESULT_FOLDER']
if config['win']:
    DATA_PATH = rf'{DATA_PATH}'
    RESULT_FOLDER = rf'{RESULT_FOLDER}'

hyp_params = {'batch': config['BATCHSIZE'],
               'lr': config['lr'], 
               'epochs': config['epochs'],
                  }

################ NNI ###########################

# optimized_params = nni.get_next_parameter()
# hyp_params.update(optimized_params)

##################### NNI ########################

epochs = hyp_params['epochs']
VIZ_epoch_period = epochs #epochs/2
BATCHSIZE = hyp_params['batch']
LR = round( hyp_params['lr'], 5)
STEP_SIZE = epochs/4
EXPECTED_IMG_SIZE = config['EXPECTED_IMG_SIZE']
EXPECTED_SIM_SIZE = config['EXPECTED_SIM_SIZE']



def main(bz, num_epochs=100, result_folder = RESULT_FOLDER, folder_description = ""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Specify Unique Directories for result
    print('-'*15, 'CHECK RESULT DIRECTORY', '-'*15)
    result_folder = result_folder+get_time_YYYYMMDDHH()+'_'+folder_description
    ensure_directory_exists(result_folder)

    # Define the architecture of the branches and trunk network
    branch1_dim = [EXPECTED_IMG_SIZE[1]*EXPECTED_IMG_SIZE[0], 100, 100, 64]  # Geometry branch dimensions (flattened image input followed by layers)
    branch2_dim = [2, 32, 32, 64]  # Source location branch
    trunk_dim = [2, 100, 100, 64]  # Trunk network (grid coordinates)

    # Define geometry_dim and output_dim based on your data
    geometry_dim = EXPECTED_IMG_SIZE  # Image dimensions (height, width)
    output_dim = EXPECTED_SIM_SIZE[0] * EXPECTED_SIM_SIZE[1]  # Simulation dimensions (pressure map height and width) #162 * 512

    # Initialize model and move it to the device (GPU/CPU)
    model = opnn(branch2_dim, trunk_dim, geometry_dim).to(device) # for CNN
    total_params = sum(p.numel() for p in model.parameters())
    print("total parameters")
    print(total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('TOTAL TRAINABLE PARAMS')
    print(trainable_params)

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    #scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=0.5)  # Reduce LR every 500 steps
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)


    # Prepare data
    dataloader_train, dataloader_valid,dataloader_test = load_data_by_split(DATA_PATH, bz)

    # Train the model
    print('-'*15, 'TRAIN', '-'*15)
    trainer = Trainer(model, optimizer, device, num_epochs, model_type='opnn')
    trainer.set_result_path(result_folder)
    model = trainer.train(dataloader_train, dataloader_valid, dataloader_test, scheduler)

    #Plot Losses
    file_paths = [trainer.train_log_path,trainer.val_log_path]
    plot_logs(file_paths, output_image=os.path.join(trainer.result_folder, "loss_plot.png"))

    
if __name__ == "__main__":
    print(f"CONFIG: DATA_PATH: {DATA_PATH}, RESULT_FOLDER: {RESULT_FOLDER}, epochs: {epochs}, VIZ_epoch_period: {VIZ_epoch_period}, BATCHSIZE: {BATCHSIZE}, STEP_SIZE: {STEP_SIZE}, EXPECTED_IMG_SIZE: {EXPECTED_IMG_SIZE}, EXPECTED_SIM_SIZE: {EXPECTED_SIM_SIZE}")
    parser = argparse.ArgumentParser(description="Experiment id or brief description, no space or slash allowed. Good Example: high_resolution_1.")
    parser.add_argument('--exp_description', type=str, default="", help='Optional description.')
    args = parser.parse_args()

    # Call the main function and pass the argument
    main(bz=BATCHSIZE,num_epochs=epochs, result_folder=RESULT_FOLDER, folder_description=args.exp_description)
    
