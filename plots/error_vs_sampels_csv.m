% How to use: drag and drop the results from the folder results you want 
% to plot in the workspac
clear all
close all
type = "linear";
num_positions = 9;
samples = round(logspace(log10(300), log10(3000), 10));

num_seeds = 3;

num_col = length(samples);
tot_data = zeros(num_seeds*num_positions, num_col);

for ii = 1:num_positions
    filename = "../results/" + type + "_results_pos_" + ii + ".csv";
    loaded_data = readmatrix(filename);
    start_idx = ((ii-1)*num_seeds+1);
    finish_idx = start_idx+num_seeds-1;
    tot_data(start_idx:finish_idx,:) = loaded_data;
end
% Example: create x values
x = samples;  % or whatever your x-axis is
avg_data = mean(tot_data,1);
std_data = std(tot_data,1);
% Calculate upper and lower bounds for confidence interval
upper = avg_data + std_data/2;
lower = avg_data - std_data/2;

% Combine into one matrix
data = [x(:), avg_data(:), upper(:), lower(:)];

% 


nGroups = size(tot_data,1)/num_seeds;
avg_data = zeros(nGroups, num_col);

for i = 1:nGroups
    rows = (i-1)*num_seeds + (1:num_seeds);
    avg_data(i,:) = mean(tot_data(rows,:), 1);
end

figure
colororder(turbo(9))  % one distinct color per line

plot(samples, avg_data.', '-o', 'LineWidth', 1.5)
xlabel('Number of samples')
ylabel('Error [dB]')
title('Average Error (seeds+config.): '+type+' model')
legend(arrayfun(@(i) "Position " + i, 1:num_positions, ...
       'UniformOutput', false), 'Location', 'best')
grid on

% Save to CSV
csvwrite(type+"_overleaf.csv", data);