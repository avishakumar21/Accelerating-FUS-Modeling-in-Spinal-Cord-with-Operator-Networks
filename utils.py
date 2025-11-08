import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms



# Helper function to log loss after each epoch
def log_loss(loss, temp_file="temp_loss_log.txt"):
    """
    Logs the loss of the current epoch to a temporary file.
    """
    with open(temp_file, "a") as f:
        f.write(f"{loss}\n")

def save_loss_to_dated_file(data_train, epochs_stop, temp_file="temp_loss_log.txt", final_dir="loss_logs"):
    """
    Reads the losses from the temporary file and saves them to a JSON file named with the
    date and time the training is completed.

    Parameters:
    data_train (str): A string representing the training data.
    epochs_stop (int): The epoch number where the training stopped.
    temp_file (str): The path to the temporary file where losses are logged.
    final_dir (str): The directory where the final JSON log file will be saved.
    """
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
    
    # Get current date and time for the file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_file = os.path.join(final_dir, f"loss_log_{timestamp}.json")

    try:
        # Read the losses from the temp file
        with open(temp_file, "r") as temp_f:
            losses = [float(line.strip()) for line in temp_f.readlines()]

        # Create the dictionary to store in the JSON file
        loss_data = {
            "data_train": data_train,
            "epochs_stop": epochs_stop,
            "losses": losses
        }

        # Write the data to a JSON file
        with open(final_file, "w") as final_f:
            json.dump(loss_data, final_f, indent=4)

    except FileNotFoundError:
        print(f"Temporary loss file '{temp_file}' not found.")
    
    # Clean up the temporary file
    os.remove(temp_file)


def ensure_directory_exists(directory_path):
    """
    Check if the given directory exists, and if not, create it.

    Parameters:
    - directory_path: str, path of the directory to check/create.
    """
    if not os.path.exists(directory_path):
        # Create the directory if it doesn't exist
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def get_time_YYYYMMDDHH():
    # Get the current date and time
    start_time = datetime.now()

    # Format the date and time as "YYYYMMDDHH"
    formatted_time = start_time.strftime('%Y%m%d%H')
    return formatted_time



## -----------------  GRAPHS ------------------------
import matplotlib.pyplot as plt

def plot_logs(file_paths, output_image="loss_plot.png"):
    """
    Reads values from each text file and plots them on a single graph.

    Parameters:
    - file_paths: list of strings, each string is a path to a txt file with logged values.
    - output_image: string, the path to the output image file (default: 'loss_plot.png').
    """
    plt.figure(figsize=(10, 6))  # Set the figure size
    
    for file_path in file_paths:
        # Read values from each txt file
        with open(file_path, 'r') as file:
            values = [float(line.strip()) for line in file.readlines()]

        # Plot the values on the graph
        plt.plot(values, label=file_path)  # Use file path as the label for each line

    # Adding labels and title to the plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training/Validation Losses')
    plt.legend(loc="upper right")  # Show the legend

    # Save the figure to a file
    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")

    # Optionally show the plot
    # plt.show()
    plt.close()

class SegmentationVisualizer:
    def __init__(self, color_map='jet'):
        self.color_map = color_map
        self.global_min = 0
        self.global_max = 1
    def minmax_normalize(self, tensor):
        """
        Normalize a tensor to the range [0, 1] using min-max normalization.
        """
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val)

    def process_image(self, image, normalize=True):
        if normalize:
            image = self.minmax_normalize(image)
        return image.permute(1, 2, 0).cpu().numpy()
        
    def process_simulation(self, simulation, normalize=True):
        if normalize:
            simulation = self.minmax_normalize(simulation)
        simulation = simulation.cpu().numpy()
        if len(simulation.shape)<2:
            return None
        return simulation

    def process_segmentation(self, segmentation, normalize=True):
        if normalize:
            segmentation = self.minmax_normalize(segmentation)
        return segmentation.cpu().numpy()
    
    


    def plot(self, image, segmentation, prediction, shared_colorbar_ax=None):
        # Convert tensors to numpy arrays for visualization
        image_np = self.process_image(image)
        segmentation_np = self.process_segmentation(segmentation)
        #simulation_np 
        prediction_np = self.process_simulation(prediction)
        if prediction_np is None:
            prediction_np = np.zeros(segmentation_np.shape)

        # Create figure for displaying all plots
        fig, ax = plt.subplots(2, 3, figsize=(15, 6))

        # Plot original input image
        ax[0, 0].imshow(image_np)
        ax[0, 0].set_title('Original Image')

        # Plot simulation on image
        ax[0, 1].imshow(image_np)
        ax[0, 1].imshow(segmentation_np, cmap=self.color_map, alpha=0.5, vmin=self.global_min, vmax=self.global_max)
        ax[0, 1].set_title('Pressure Simulation on Image')

        # Plot prediction on image
        ax[0, 2].imshow(image_np)
        ax[0, 2].imshow(prediction_np, cmap=self.color_map, alpha=0.5,vmin=self.global_min, vmax=self.global_max)
        ax[0, 2].set_title('Pressure Prediction on Image')

        # Plot simulation
        seg_plot =ax[1, 0].imshow(segmentation_np, cmap=self.color_map,vmin=self.global_min, vmax=self.global_max)
        ax[1, 0].set_title('Simulation')

        # Plot prediction
        ax[1, 1].imshow(prediction_np, cmap=self.color_map,vmin=self.global_min, vmax=self.global_max)
        ax[1, 1].set_title('Prediction')

        # Plot prediction - simulation
        diff = np.abs(prediction_np - segmentation_np)
        ax[1, 2].imshow(diff, cmap=self.color_map, vmin=self.global_min, vmax=self.global_max)
        ax[1, 2].set_title('Prediction - Simulation')

        if shared_colorbar_ax is None:
            # Create a color axis that spans across the entire figure
            cbar_ax = fig.add_axes([0.99, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(seg_plot, ax=ax.ravel().tolist(), orientation='vertical',cax=cbar_ax)
            cbar.set_label('Pressure/Prediction Intensity')

        #plt.tight_layout()
        #plt.show()
        plt.close()
        return fig

    def visualize_batch(self, images, simulation, predictions, batch, comment, result_folder = ''):
        # Loop through batch and plot each image/segmentation/label/prediction set
        for i in range(images.shape[0]):
            #resize image
            #print(simulation[i].shape[-2], simulation[i].shape[-1],'img',images[i].shape )
            # resize_transform = transforms.Compose([
            #     transforms.Resize((simulation[i].shape[-2], simulation[i].shape[-1]))  # Resize to sim shape
            # ])
            # resized_image = resize_transform(images[i])
            fig = self.plot(images[i], simulation[i], predictions[i])
            fig.savefig(os.path.join(result_folder, f'{comment}sample_bz_{batch}_{i}.png') )
            plt.close()

def plot_prediction(image, simulation, prediction, batch, comment = '', result_folder=''):
    visualizer = SegmentationVisualizer()
    images_01 = visualizer.minmax_normalize(image)
    simulations_01 = visualizer.minmax_normalize(simulation)
    predictions_01 = visualizer.minmax_normalize(prediction)
    #print(images_01.shape,simulations_01.shape,predictions_01.shape)
    visualizer.visualize_batch(images_01, simulations_01, predictions_01, batch, comment = comment,result_folder=result_folder)



## -----------------  Storage ------------------------
def store_model(model,optimizer,epoch, result_path):
    checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,  # if resume from a specific epoch
    }
    
    torch.save(checkpoint, os.path.join(result_path, 'model_checkpoint.pth'))