import time
import numpy as np

# build a big graph (normalized weight matrix)
def buildGraph(MatX, rbf_sigma = None):
    num_samples = MatX.shape[0]
    affinity_matrix = np.zeros((num_samples, num_samples), np.float32)
    if rbf_sigma == None:
        raise ValueError('You should input a sigma of rbf kernel!')
    for i in range(num_samples):
        row_sum = 0.0
        for j in range(num_samples):
            diff = MatX[i, :] - MatX[j, :]
            affinity_matrix[i][j] = np.exp(sum(diff**2) / (-2.0 * rbf_sigma**2))
            row_sum += affinity_matrix[i][j]
        affinity_matrix[i][:] /= row_sum
    return affinity_matrix


# label propagation
def labelPropagation(Mat_Label, Mat_Unlabel, labels, rbf_sigma = 1.5, \
                    max_iter = 500, tol = 1e-4):
    # initialize
    num_label_samples = len(Mat_Label)
    num_unlabel_samples = len(Mat_Unlabel)
    num_samples = num_label_samples + num_unlabel_samples
    labels_list = np.unique(labels)
    num_classes = len(labels_list)
    
    MatX = np.vstack((Mat_Label, Mat_Unlabel))
    clamp_data_label = np.zeros((num_label_samples, num_classes), np.float32)
    for i in range(num_label_samples):
        #print(labels[i])
        j = int(labels[i]) - 1
        clamp_data_label[i][j] = 1.0
    
    label_function = np.zeros((num_samples, num_classes), np.float32)
    label_function[0 : num_label_samples] = clamp_data_label
    label_function[num_label_samples : num_samples] = -1
    
    # graph construction
    affinity_matrix = buildGraph(MatX, rbf_sigma)
    
    # start to propagation
    iter = 0; pre_label_function = np.zeros((num_samples, num_classes), np.float32)
    changed = np.abs(pre_label_function - label_function).sum()
    while iter < max_iter and changed > tol:
        if iter % 1 == 0:
            print ("---> Iteration %d/%d, changed: %f" % (iter, max_iter, changed))
        pre_label_function = label_function
        iter += 1
        
        # propagation
        label_function = np.dot(affinity_matrix, label_function)
        
        # clamp
        label_function[0 : num_label_samples] = clamp_data_label
        
        # check converge
        changed = np.abs(pre_label_function - label_function).sum()
    
    # get terminate label of unlabeled data
    unlabel_data_labels = np.zeros(num_unlabel_samples)
    for i in range(num_unlabel_samples):
        unlabel_data_labels[i] = np.argmax(label_function[i+num_label_samples])
    
    return unlabel_data_labels