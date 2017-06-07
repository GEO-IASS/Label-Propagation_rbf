function [affinity_matrix] = buildGraph(MatX, rbf_sigma)
% build a big graph (normalized weight matrix)  
num_samples = size(MatX,1);
affinity_matrix = zeros(num_samples, num_samples, 'double');
for i = 1:num_samples
    row_sum = 0.0;
    for j = 1:num_samples
        diff = MatX(i, :) - MatX(j, :);
        affinity_matrix(i, j) = exp(sum(diff .^ 2) / (-2.0 * rbf_sigma .^ 2));
        row_sum = row_sum + affinity_matrix(i , j);
    end
    affinity_matrix(i , :) = affinity_matrix(i , :) / row_sum;
end
end