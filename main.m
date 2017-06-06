function [] = main()
[Mat_Label, Mat_Unlabel, labels] = loadDataFromTxt();
rbf_sigma = 1.5;
max_iter = 100000;
tic
unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, rbf_sigma, max_iter);
toc
show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels);
end
