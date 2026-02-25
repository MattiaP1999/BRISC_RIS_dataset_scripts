import numpy as np
from scipy.linalg import khatri_rao
from scipy.io import loadmat
import matplotlib.pyplot as plt
import h5py
import pickle as pkl
import gzip
import my_lib as my_lib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset

# Define the regression neural network architecture
class RegressionNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def trainModel(
    model,
    epochs,
    lr,
    train_loader,
    val_loader,
    len_train_dataset,
    len_val_dataset,
    early_stopping_patience=20
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # L2 regularization strength

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len_train_dataset

        # ---- Validation ----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len_val_dataset

        # ---- Scheduler ----
        scheduler.step(val_loss)

        # ---- Logging ----
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, "
            f"Val Loss={val_loss:.4f}, "
            f"Current LR={current_lr:.4e}"
        )

        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}. "
                f"Best val loss: {best_val_loss:.4f}"
            )
            break

    # ---- Restore Best Model ----
    model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


num_sbc = 242 # Set the number of subcarriers
perform_permutation = True
positions = [1,2,3,4,5,6,7,8,9]
for position in positions: #
    #filename = r"C:\Users\mattia\Desktop\linearity verification\data\antennaLog_pos5.mat"
    data_file = f"data/antennaLog_pos{position}.mat" # Replace with your actual data file path
    conf_file= f"data/configurations_pos_{position}.txt"# Replace with your actual config file path
    with h5py.File(data_file, 'r') as f:
        csi = f['csi'][:]
    parse_csi = csi['real'] + 1j * csi['imag']
    parse_csi = parse_csi.transpose() # Transpose to get correct shape
    configurations = []
    with open(conf_file, "r") as f:
        for line in f:
            configurations.append(line.strip())

    configurations = np.array(configurations)
    subcarrier_list = np.arange(0, num_sbc, 1, dtype=np.int32)
    num_sbc_selected = len(subcarrier_list)
    parse_csi = parse_csi[subcarrier_list,:]

    unique_configs, idx = np.unique(configurations, return_index=True)
    ordered_unique_configs = unique_configs[np.argsort(idx)]
    conf_from_data = []
    for idx,conf in enumerate(ordered_unique_configs):
        matrix_converted = my_lib.hex_to_matrix(conf, N=16)
        #if(idx>520 and idx<536):
            #display_matrix(matrix_converted,idx)
        conf_from_data.append(matrix_converted.flatten('F')) # check if 'F' order is needed

    conf_from_data = np.array(conf_from_data)
    conf_from_data = conf_from_data.transpose() # check if transpose is needed
    tot_conf = np.int32(unique_configs.shape[0])
    channels = []
    # Average channels corresponding to the same configuration
    channels_np = np.zeros((num_sbc_selected, tot_conf), dtype=complex)
    for i in range(tot_conf):
        indexes_conf = np.where(configurations== ordered_unique_configs[i])[0]
        current_csi_target = parse_csi[:,indexes_conf]
        ch_t_mean = np.mean(current_csi_target, axis=1)
        channels_np[:,i] = ch_t_mean
    print(channels_np.shape)

    # Data normalization
    mean_ch = np.mean(channels_np, axis=1, keepdims=True)
    var_ch = np.var(channels_np, axis=1, keepdims=True)
    channels_np = channels_np - mean_ch
    channels_np = channels_np / np.sqrt(var_ch)

    channels_np_concat = np.vstack((channels_np.real, channels_np.imag))

    # Random shuffle conf_from_data and channels_np_concat # except the last n_test samples
    n_test = 3000
    num_seeds = 3
    n_val = 1500  
    indices = np.arange(conf_from_data.shape[1] - n_test)
    if perform_permutation:
        np.random.shuffle(indices)
    tot_indices = np.concatenate((indices, np.arange(conf_from_data.shape[1] - n_test, conf_from_data.shape[1])))
    conf_from_data = conf_from_data[:,tot_indices]
    channels_np_concat = channels_np_concat[:,tot_indices]

    X = torch.from_numpy(conf_from_data.T).float()   # (samples, features)
    y = torch.from_numpy(channels_np_concat.T).float()   # (samples, outputs)
    dataset = TensorDataset(X, y)


    train_samples_list = np.logspace(np.log10(300), np.log10(3000), 10, dtype=int)
    n_total = len(dataset)
    mean_errors_db = np.zeros((num_seeds, len(train_samples_list)))

    for idx_seed in range(num_seeds):
        torch.manual_seed(41+idx_seed)
        np.random.seed(41+idx_seed)
        for idx_n_train, n_train in enumerate(train_samples_list):
            train_indices = list(range(0, n_train))
            val_indices   = list(range(n_train, n_train + n_val))
            test_indices  = list(range(n_total - n_test, n_total))

            train_dataset = Subset(dataset, train_indices)
            val_dataset   = Subset(dataset, val_indices)
            test_dataset  = Subset(dataset, test_indices)

            # Dimensions
            print("Running simulation with seed:", idx, "and training samples:", n_train)
            print("Total number of configurations:", n_total)
            print("Number of training samples:", len(train_dataset))
            print("Number of validation samples:", len(val_dataset))
            print("Number of test samples:", len(test_dataset))

            lr_candidates =[1e-4, 1e-3] 
            batch_size_candidates = [2,4,16]
            epochs = 300
            best_val_loss = float("inf")
            best_model_state = None
            best_config = None
            for lr in lr_candidates:
                for batch_size in batch_size_candidates:
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True
                    )

                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=batch_size,
                        shuffle=False
                    )

                    model = RegressionNN(
                        input_dim=X.shape[1],
                        output_dim=y.shape[1]
                    )

                    print(f"Training with initial lr={lr}, batch_size={batch_size}")

                    model, train_losses, val_losses = trainModel(
                        model,
                        epochs=epochs,
                        lr=lr,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        len_train_dataset=len(train_dataset),
                        len_val_dataset=len(val_dataset)
                    )

                    final_val_loss = val_losses[-1]

                    if final_val_loss < best_val_loss:
                        best_val_loss = final_val_loss
                        best_model_state = model.state_dict().copy()
                        best_config = (lr, batch_size)

            best_model = RegressionNN(
                input_dim=X.shape[1],
                output_dim=y.shape[1]
            )
            best_model.load_state_dict(best_model_state)
            print("Best config:", best_config)
            print("Best validation loss:", best_val_loss)

            best_model.eval()
            predictions = []
            preds = []
            trues = []
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False
            )
            with torch.no_grad():
                for xb, yb in test_loader:
                    preds.append(best_model(xb).cpu())
                    trues.append(yb.cpu())

            y_pred = torch.cat(preds, dim=0)   # (n_test, 2*n_channels)
            y_true = torch.cat(trues, dim=0)   # (n_test, 2*n_channels)

            y_pred_np = y_pred.numpy()
            y_true_np = y_true.numpy()

            n_channels = y_pred_np.shape[1] // 2

            real_part = y_pred_np[:, :n_channels]
            imag_part = y_pred_np[:, n_channels:]

            channels_pred_complex = real_part + 1j * imag_part
            channels_pred_complex = channels_pred_complex.T
            channels_pred_complex = channels_pred_complex * np.sqrt(var_ch) + mean_ch # Undo the normalization
            print("Predicted channels shape:", channels_pred_complex.shape)

            n_channels = y_true_np.shape[1] // 2

            real_part = y_true_np[:, :n_channels]
            imag_part = y_true_np[:, n_channels:]

            channels_true_complex = real_part + 1j * imag_part
            channels_true_complex = channels_true_complex.T
            channels_true_complex = channels_true_complex * np.sqrt(var_ch) + mean_ch # Undo the normalization 
            print("True channels shape:", channels_true_complex.shape)


            num_meas_test = channels_true_complex.shape[1]
            squared_norm_error = np.linalg.norm(channels_pred_complex-channels_true_complex)**2/num_meas_test
            average_ch_measured = np.linalg.norm(channels_true_complex)**2/num_meas_test
            mean_error = 10*np.log10(squared_norm_error/average_ch_measured)
            print(f"Mean relative error (dB) over test set: {mean_error:.4f} dB")
            mean_errors_db[idx_seed, idx_n_train] = mean_error
            np.savetxt(f"./results/nn_results_extra_pos_{position}.csv", mean_errors_db, delimiter=",")