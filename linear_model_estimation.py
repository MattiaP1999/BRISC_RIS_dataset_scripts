import numpy as np
from scipy.linalg import khatri_rao
from scipy.io import loadmat
import h5py
import pickle as pkl
import gzip
import my_lib as my_lib


# Set current directory
use_bias = False # Set to True if want to include bias (LMB)
num_sbc = 242 # Number of subcarrier to select
perform_permutation = True # Set to True if want to perform permutation of the dataset
positions = [1,2,3,4,5,6,7,8,9]

for position in positions:
    # Load data
    data_file = f"../RIS_Dataset_UNIPD_UNIBS/data/antennaLog_pos{position}.mat" # Replace with your actual data file path
    conf_file= f"../RIS_Dataset_UNIPD_UNIBS/data/configurations_pos_{position}.txt" # Replace with your actual config file path

    with h5py.File(data_file, 'r') as f:
        csi = f['csi'][:]
    parse_csi = csi['real'] + 1j * csi['imag']
    parse_csi = parse_csi.transpose() # Transpose to get correct shape

    # Get configurations
    configurations = []
    with open(conf_file, "r") as f:
        for line in f:
            configurations.append(line.strip())

    configurations = np.array(configurations)
    subcarrier_list = np.arange(0, num_sbc, 1, dtype=np.int32)
    num_sbc_selected = len(subcarrier_list)
    parse_csi = parse_csi[subcarrier_list,:]

    # Select unique configurations
    unique_configs, idx = np.unique(configurations, return_index=True)
    ordered_unique_configs = unique_configs[np.argsort(idx)]
    conf_from_data = []
    for idx,conf in enumerate(ordered_unique_configs):
        matrix_converted = my_lib.hex_to_matrix(conf, N=16)
        # Uncomment to plot desired RIS configurations
        #if(idx>520 and idx<536):
            #display_matrix(matrix_converted,idx)
        conf_from_data.append(matrix_converted.flatten('F')) # check if 'F' order is needed

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

    # Random shuffle conf_from_data and channels_np_concat # except the last n_test samples
    n_test = 3000 # Number of test samples
    num_seeds = 3 # Number of random seeds 
    n_val = 1500 # Number of validation samples
    indices = np.arange(conf_from_data.shape[1] - n_test)
    if perform_permutation:
        np.random.shuffle(indices)
    tot_indices = np.concatenate((indices, np.arange(conf_from_data.shape[1] - n_test, conf_from_data.shape[1])))
    conf_from_data = conf_from_data[:,tot_indices]
    channels_np_concat = channels_np[:,tot_indices]
    train_samples_list = np.logspace(np.log10(300), np.log10(3000), 10, dtype=int) # Number of training samples
    n_total = channels_np.shape[1]
    mean_errors_db = np.zeros((num_seeds, len(train_samples_list)))
    for idx_seed in range(num_seeds):
        print(f"Running {idx_seed} simulation:")
        for idx_n_train, n_train in enumerate(train_samples_list):
            train_indices = list(range(0, n_train))
            test_indices  = list(range(n_total - n_test, n_total))
            conf_fit_train = 1-conf_from_data[:,train_indices]
            conf_fit_test = 1-conf_from_data[:,test_indices]
            ch_meas_fit = channels_np_concat[:,train_indices]
            ch_meas_test = channels_np_concat[:,test_indices]

            print("Total number of configurations:", n_total)
            print("Running simulation with fitting samples:", conf_fit_train.shape[1])
            print("Running simulation with testing samples:", conf_fit_test.shape[1])

            theta = np.deg2rad([92]) # Phase shift of the RIS elements in degrees
            amp = 0.4 # Amplitude of the RIS elements
            pred_channels = np.zeros_like(ch_meas_test)
            E_tot = np.zeros((num_sbc_selected,256),dtype=complex) # Matrix to store the estimated E (Katri-rao product) for each subcarrier

            coeff_train = np.exp(1j*conf_fit_train*theta)
            coeff_train[conf_fit_train != 0] *= amp
            coeff_test = np.exp(1j*conf_fit_test*theta)
            coeff_test[conf_fit_test != 0] *= amp
            if not use_bias:
                E_tot = my_lib.estimate_katri_E_full(ch_meas_fit,coeff_train)
                ch_est_test = E_tot@coeff_test
            else:
                E_tot,bias = my_lib.estimate_katri_E_full_bias(ch_meas_fit,coeff_train)
                ch_est_test = E_tot@coeff_test+bias.reshape(-1,1)
            num_meas_test = ch_meas_test.shape[1]
            squared_norm_error = np.linalg.norm(ch_est_test-ch_meas_test)**2/num_meas_test
            average_ch_measured = np.linalg.norm(ch_meas_test)**2/num_meas_test
            mean_error = 10*np.log10(squared_norm_error/average_ch_measured)

            print(f"Mean relative error (dB) over test set: {mean_error:.4f} dB")
            mean_errors_db[idx_seed,idx_n_train] = mean_error
            if use_bias:
                np.savetxt(f"./results/linear_bias_results_pos_{position}.csv", mean_errors_db, delimiter=",")
            else:
                np.savetxt(f"./results/linear_results_pos_{position}.csv", mean_errors_db, delimiter=",")

# Save mean error to csv file
