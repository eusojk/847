% USPS.mat from https://github.com/jiayuzhou/CSE847/blob/master/data/USPS.mat?raw=true

% Loading the matrix:
dataset = load("USPS.mat");
A = dataset.A


% PART 1
% apply to the data using p = 10, 50, 100, 200 principal components:
[dir_10, pca_10]   = apply_pca(A, 10);
[dir_50, pca_50]   = apply_pca(A, 50);
[dir_100, pca_100] = apply_pca(A, 100);
[dir_200, pca_200] = apply_pca(A, 200);


% PART 2
% Reconstructing the image
mu = mean(A);
reconstructed_pca_10  = reconstruct_data(pca_10, dir_10, mu); 
reconstructed_pca_50  = reconstruct_data(pca_50, dir_50, mu); 
reconstructed_pca_100 = reconstruct_data(pca_100, dir_100, mu); 
reconstructed_pca_200 = reconstruct_data(pca_200, dir_200, mu); 

% The total reconstruction errors for p = 10, 50, 100, 200.
error_10  = reconstruction_error(A, reconstructed_pca_10)
error_50  = reconstruction_error(A, reconstructed_pca_50)
error_100 = reconstruction_error(A, reconstructed_pca_100)
error_200 = reconstruction_error(A, reconstructed_pca_200)

% Example of plotting the (the first two) of the reconstructed images for p = 10
A10_1 = reshape(reconstructed_pca_10(1,:), 16, 16);
A10_2 = reshape(reconstructed_pca_10(2,:), 16, 16);
imshow(A10_1')
imshow(A10_2')

% FUNCTIONS for PART 1 & 2

function output = reconstruct_data(pca_components, pca_vectors, mu)
% This functions reconstructs the original dataset given the PCA values
pca_data = pca_components * pca_vectors';
output = bsxfun(@plus, pca_data, mu);
end


function [directions, components] = apply_pca(matrix_input, dimension)
% This function implements PCA using SVD
% Inputs:
%   - matrix_input: is the raw data
%   - dimension: to reducd the raw data to
% Outputs:
%   - directions: top k principal directions v1, v2, ..., vk
%   - components: corresponding PCs 
centered_data = matrix_input - repmat(mean(matrix_input, 1), size(matrix_input, 1), 1);
[~, ~, V] = svds(cov(centered_data), dimension);
components = centered_data * V;
directions = V;
end

function error = reconstruction_error(data_original, data_reconstructed)
diff = abs(data_original - data_reconstructed).^2;
error = sum(diff(:))/numel(data_original);
end
