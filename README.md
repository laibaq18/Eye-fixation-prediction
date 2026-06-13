Go to the folder where scripts and data is:

1) Create virtual environment and install dependencies
  
2) Train via: 
``python3 train.py --data-root ../cv2_project_data --epochs 5 --batch-size 4 --freeze-backbone``

3) Create predictions via:
``python3 predict_test.py   --data-root ../cv2_project_data   --checkpoint checkpoints/best.pt   --save-dir test_predictions``


Input image : [B, 3, 224, 224] -> FCN ResNet-50 -> Raw logits : [B, 1, 224, 224] -> Gaussian smoothing : [B, 1, 224, 224] -> Add log center bias : [B, 1, 224, 224] -> Final fixation logits : [B, 1, 224, 224]
