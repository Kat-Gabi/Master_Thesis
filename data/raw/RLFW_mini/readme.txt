RFW_test:
Racial Faces in-the-Wild (RFW) database can be used to fairly measure performance of different races in deep face recognition. It consists of four testing subsets, namely Caucasian, Asian, Indian and African. Each subset contains about 10K images of 3K individuals for face verification.

Note£ºThe images of our RFW are carefully selected from MS-Celeb-1M, it is meaningless to utilize our testing set to evaluate the models trained with MS-Celeb-1M. If you really want to do it, we recommend deleting the overlapping images in MS-Celeb-1M according to our file, i.e. test_people.txt, or you can directly use the training set, i.e. MS1M_wo_RFW, provided by us.

We provide two type of formats of images. One is '.jpg' which can be found in the folder named 'images'. The other is '.rec' or '.bin' which can be found in the folder names 'rec_for_mxnet' or 'bin_for_mxnet'. When you use mxnet to implement face recognition network, such as the code of insightface (Arcface: Additive angular margin loss for deep face recognition), you can directly use '.rec' and '.bin' to train or test model. 

For '.jpg', we provide loosely cropped faces for testing. Each identity is named as '< freebaseID >' provided by freebase, e.g. 'm.0xnkj'. Each face image has unique name, e.g. 'm.0xnkj/m.0xnkj_00000.jpg' where 'm.0xnkj_00000.jpg' is named in a way '< identityID >_< faceID >.jpg'. 

Testing:
Test Data_v1.
Test_lmk_v1. Estimated 5 facial landmarks on the provided loosely cropped faces.
Test_images_v1. The testing image list and label, e.g., 'm.0xnkj_ 0002.jpg 0'
Test_pairs_v1. 10 disjoint splits of image pairs are given, and each contains 300 positive pairs and 300 negative pairs similar to LFW.
Test_people_v1. The overlapped identities between RFW and MS-Celeb-1M and the number of images per identity.

Please cite the following if you make use of the datasets:
[1] Mei Wang, Weihong Deng, Jiani Hu, Xunqiang Tao, Yaohai Huang. Racial Faces in the Wild: Reducing Racial Bias by Information Maximization Adaptation Network.
[2] Mei Wang, Weihong Deng. Mitigate Bias in Face Recognition using Skewness-Aware Reinforcement Learning. 
[3] Mei Wang, Weihong Deng. Deep face recognition: A Survey. 



