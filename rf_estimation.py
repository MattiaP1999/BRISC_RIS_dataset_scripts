from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import numpy as np
from scipy.linalg import khatri_rao
from scipy.io import loadmat
import h5py
import my_lib as my_lib
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import random_split


# Set current directory

num_sbc = 242 # Number of subcarrier to select
perform_permutation = True
positions = [1,2,3,4,5,6,7,8,9]

for position in positions:
    # Load data
    data_file = f"../RIS_Dataset_UNIPD_UNIBS/data/antennaLog_pos{position}.mat" # Replace with your actual data file path
    conf_file= f"../RIS_Dataset_UNIPD_UNIBS/data/configurations_pos_{position}.txt" # Replace with your actual config file path
    with h5py.File(data_file, 'r') as f:
        csi = f['csi'][:]
    parse_csi = csi['real'] + 1j * csi['imag']
    parse_csi = parse_csi.transpose() # Transpose to get correct shape

    # Load configurations
    configurations = []
    with open(conf_file, "r") as f:
        for line in f:
            configurations.append(line.strip())

    configurations = np.array(configurations)

    subcarrier_list = np.arange(0, num_sbc, 1, dtype=np.int32)
    num_sbc_selected = len(subcarrier_list)
    parse_csi = parse_csi[subcarrier_list,:]

    # Get unique configurations
    unique_configs, idx = np.unique(configurations, return_index=True)
    ordered_unique_configs = unique_configs[np.argsort(idx)]
    conf_from_data = []
    for idx,conf in enumerate(ordered_unique_configs):
        matrix_converted = my_lib.hex_to_matrix(conf, N=16)
        conf_from_data.append(matrix_converted.flatten('F'))

    conf_from_data = np.array(conf_from_data)
    conf_from_data = conf_from_data.transpose()

    # Average the CSI over the measurements corresponding to the same configuration
    tot_conf = np.int32(unique_configs.shape[0])
    channels = []
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

    n_test = 3000 # Number of test samples
    n_val = 1500 # Number of validation samples
    num_seeds = 3 # Number of random seeds

    # Random shuffle conf_from_data and channels_np_concat except the last n_test samples
    indices = np.arange(conf_from_data.shape[1] - n_test)
    if perform_permutation:
        np.random.shuffle(indices)
    tot_indices = np.concatenate((indices, np.arange(conf_from_data.shape[1] - n_test, conf_from_data.shape[1])))
    conf_from_data = conf_from_data[:,tot_indices]
    channels_np_concat = channels_np_concat[:,tot_indices]

    # Convert to PyTorch tensors
    X = torch.from_numpy(conf_from_data.T).float()   # (samples, features)
    y = torch.from_numpy(channels_np_concat.T).float()   # (samples, outputs)
    dataset = TensorDataset(X, y)


    train_samples_list = np.logspace(np.log10(300), np.log10(3000), 10, dtype=int)
    n_total = len(dataset)
    mean_errors_db = np.zeros((num_seeds, len(train_samples_list)))
    for idx_seed in range(num_seeds):
        torch.manual_seed(41 + idx_seed)
        np.random.seed(41 + idx_seed)
        for idx_n_train, n_train in enumerate(train_samples_list):
            train_indices = list(range(0, n_train)) # Training indices
            val_indices   = list(range(n_train, n_train + n_val)) # Validation indices
            test_indices  = list(range(n_total - n_test, n_total)) # Test indices (last n_test)
            # Extract NumPy data
            X_train = X[train_indices]
            y_train = y[train_indices]

            X_val = X[val_indices]
            y_val = y[val_indices]

            X_test = X[test_indices]
            y_test = y[test_indices]

            print("Running simulation with seed:", idx_seed,
                "and training samples:", n_train)

            # ---------------------------
            # Random Forest hyperparams
            # ---------------------------
            n_estimators_list = [30,70]
            max_depth_list = [20,30]

            best_val_loss = float("inf")
            best_model = None
            best_config = None

            for n_estimators in n_estimators_list:
                for max_depth in max_depth_list:
                    rf = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=41 + idx_seed,
                        n_jobs=-1
                    )

                    model = MultiOutputRegressor(rf)
                    model.fit(X_train, y_train)

                    y_val_pred = model.predict(X_val)
                    val_loss = np.mean(np.abs(y_val_pred - y_val.to('cpu').numpy()))

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss.copy()
                        best_model = model
                        best_config = (n_estimators, max_depth)

            print("Best RF config:", best_config)
            print("Best validation loss:", best_val_loss)

            # ---------------------------
            # Test evaluation
            # ---------------------------
            y_pred_np = best_model.predict(X_test) 
            y_true_np = y_test # Assignment for better readability

            n_channels = y_pred_np.shape[1] // 2 # Number of channels is half the number of outputs (real and imaginary parts)

            # Convert to complex representation
            real_part = y_pred_np[:, :n_channels]
            imag_part = y_pred_np[:, n_channels:]
            channels_pred_complex = (real_part + 1j * imag_part).T
            channels_pred_complex = channels_pred_complex * np.sqrt(var_ch) + mean_ch # Undo normalization

            real_part = y_true_np[:, :n_channels]
            imag_part = y_true_np[:, n_channels:]
            channels_true_complex = (real_part + 1j * imag_part).T
            channels_true_complex = channels_true_complex * np.sqrt(var_ch) + mean_ch # Undo normalization

            channels_true_complex = channels_true_complex.to('cpu').numpy()
            
            # Evaluate the normalized MSE
            num_meas_test = channels_true_complex.shape[1]
            squared_norm_error = np.linalg.norm(channels_pred_complex-channels_true_complex)**2/num_meas_test
            average_ch_measured = np.linalg.norm(channels_true_complex)**2/num_meas_test
            mean_error = 10*np.log10(squared_norm_error/average_ch_measured)
            mean_errors_db[idx_seed, idx_n_train] = mean_error
            np.savetxt(
                f"./results/rf_results_pos_{position}.csv",
                mean_errors_db,
                delimiter=","
            )