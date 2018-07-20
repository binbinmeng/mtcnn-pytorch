## mtcnn-pytorch
1. This project is for mtcnn train and testing.
2. Using python3, nvidia NGC pytorch image 18.06-py3
3. Modify PNet: add replace conv3 with deepwise convolution, acheieve accuracy=0.9348 with --lr 0.005.  
4. Training RNet with --lr 0.005 --batch_size 2048, get accuracy=0.9695


## Traing mtcnn Model Steps

MTCNN have three networks called PNet, RNet and ONet.So we should train it on three stage, and each stage depend on previous network which will generate train data to feed current train net, also propel the minimum loss between two networks. Please download the train face datasets before your training. We use WIDER FACE and CelebA .WIDER FACE is used for training face classification and face bounding box, also CelebA is used for face landmarks. The original wider face annotation file is matlab format, you must transform it to text. I have put the transformed annotation text file into anno_store/wider_origin_anno.txt. This file is related to the following parameter called --anno_file.

Create the DFace train data temporary folder, this folder is involved in the following parameter --dface_traindata_store

    mkdir {your dface traindata folder}

### 1. Traing PNet
Generate PNet Train data and annotation file

    python dface/prepare_data/gen_Pnet_train_data.py --prefix_path {annotation file image prefix path, just your local wider face images folder} --dface_traindata_store  {dface train data temporary folder you made before }  --anno_file ｛wider face original combined  annotation file, default anno_store/wider_origin_anno.txt}

Assemble annotation file and shuffle it

    python dface/prepare_data/assemble_pnet_imglist.py

Train PNet model

    python dface/train_net/train_p_net.py
   
### 2. Traing RNet
Generate RNet Train data and annotation file

    python dface/prepare_data/gen_Rnet_train_data.py --prefix_path {annotation file image prefix path, just your local wider face images folder} --dface_traindata_store {dface train data temporary folder you made before } --anno_file ｛wider face original combined  annotation file, default anno_store/wider_origin_anno.txt} --pmodel_file {your PNet model file trained before}

Assemble annotation file and shuffle it

    python dface/prepare_data/assemble_rnet_imglist.py

Train RNet model

    python dface/train_net/train_r_net.py

    
### 3. Traing ONet
Generate ONet Train data and annotation file

    python dface/prepare_data/gen_Onet_train_data.py --prefix_path {annotation file image prefix path, just your local wider face images folder} --dface_traindata_store {dface train data temporary folder you made before } --anno_file ｛wider face original combined  annotation file, default anno_store/wider_origin_anno.txt} --pmodel_file {your PNet model file trained before} --rmodel_file {your RNet model file trained before}

Generate ONet Train landmarks data (as MTCNN landmatk is a subtask,so landmark detection is not so good. you also can close it' loss in code) Meaning we can skip this step.

    python dface/prepare_data/gen_landmark_48.py

Assemble annotation file and shuffle it

    python dface/prepare_data/assemble_onet_imglist.py

Train ONet model

    python dface/train_net/train_o_net.py

Note: a suitable learning rate and batch size is very important 

## Test Face Detection

If you don't want to train,i have put onet_epoch.pt,pnet_epoch.pt,rnet_epoch.pt in model_store folder.You just try test_image.py

    python test_image.py
