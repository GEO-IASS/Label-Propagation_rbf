import time
import math
import random
import numpy as np
from label_propagation import labelPropagation

# show
def show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels): 
    import matplotlib.pyplot as plt
    for i in range(len(Mat_Label)):
        if int(labels[i]) == 0:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], '*w')
        elif int(labels[i]) == 1:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], '*b')
        elif int(labels[i]) == 2:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], '*c')
        elif int(labels[i]) == 3:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], '*g')
        elif int(labels[i]) == 4:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], '*k')
        elif int(labels[i]) == 5:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], '*m')
        elif int(labels[i]) == 6:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], '*r')
        elif int(labels[i]) == 7:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], '*y')
            
    for i in range(len(Mat_Unlabel)):
        if int(unlabel_data_labels[i]) == 0:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], '*w')
        elif int(unlabel_data_labels[i]) == 1:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], '*b')
        elif int(unlabel_data_labels[i]) == 2:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], '*c')
        elif int(unlabel_data_labels[i]) == 3:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], '*g')
        elif int(unlabel_data_labels[i]) == 4:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], '*k')
        elif int(unlabel_data_labels[i]) == 5:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], '*m')
        elif int(unlabel_data_labels[i]) == 6:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], '*r')
        elif int(unlabel_data_labels[i]) == 7:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], '*y')
        #plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], '*', unlabel_data_labels[i])  
    
    plt.xlabel('X'); plt.ylabel('Y')
    plt.show()  

def loadDataFromTxt():
    f = open('dataset.txt','r')
    f_list = f.readlines()
    length = len(f_list)
    
    total_select = math.floor(length * 0.99)
    
    indexes = []
    for t in range(0, length):
        indexes.append(t+1)
    
    selected_idxes = random.sample(indexes, total_select)
    rest_idxes = set(indexes).difference(set(selected_idxes))
    
    selected_data = []
    rest_data = []
    for s_idx in selected_idxes:
        selected_data.append(f_list[s_idx-1])
        
    for r_idx in rest_idxes:
        rest_data.append(f_list[r_idx-1])
    
    selected_matrix = string2Arr(selected_data)
    rest_matrix = string2Arr(rest_data)
    
    labeled_labels = selected_matrix[:,2]
    
    f.close()
    return selected_matrix, rest_matrix, labeled_labels

def string2Arr(stringArr):
    length = len(stringArr)
    matrix = np.array(length)
    first_ele = True
    for data in stringArr:
       data = data.strip('\n')
       nums = data.split("\t")
       if first_ele:
           for idx, x in enumerate(nums):
               if idx == 0 or idx == 1:
                   nums[idx] = float(x)
               elif idx == 2:
                   nums[idx] = int(x)
           matrix = np.array(nums)
           first_ele = False
       else:
           for idx, x in enumerate(nums):
               if idx == 0 or idx == 1:
                   nums[idx] = float(x)
               elif idx == 2:
                   nums[idx] = int(x)
           matrix = np.c_[matrix,nums]
    matrix = matrix.transpose()
    return matrix

# main function
if __name__ == "__main__":
    Mat_Label, Mat_Unlabel, labels = loadDataFromTxt()
    unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, max_iter = 1000)
    show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels)
    