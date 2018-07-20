# mtcnn-pytorch
1. This project is for mtcnn train and testing.
2. Using python3, nvidia NGC pytorch image 18.06-py3
3. Modify PNet: add replace conv3 with deepwise convolution, acheieve accuracy=0.9348 with --lr 0.005.  
4. Training RNet with --lr 0.005 --batch_size 2048, get accuracy=0.9695
