Go to the folder where scripts and data is:

1) Create virtual environment and install dependencies
  
2) Train via: 
``python3 train.py --data-root ../cv2_project_data --epochs 5 --batch-size 4 --freeze-backbone``

3) Create predictions via:
``python3 predict_test.py   --data-root ../cv2_project_data   --checkpoint checkpoints/best.pt   --save-dir test_predictions``

FCN Resnet50:
Input image : [B, 3, 224, 224] -> FCN ResNet-50 -> Raw logits : [B, 1, 224, 224] -> Gaussian smoothing : [B, 1, 224, 224] -> Add log center bias : [B, 1, 224, 224] -> Final fixation logits : [B, 1, 224, 224]


DeepGaze2 Like:
Input image : [B, 3, 224, 224] -> Frozen VGG-19 feature extractor -> Selected VGG features:
      conv5_1 : [B, 512, 14, 14] <br>
      relu5_1 : [B, 512, 14, 14] <br>
      relu5_2 : [B, 512, 14, 14] <br>
      conv5_3 : [B, 512, 14, 14] <br>
      relu5_4 : [B, 512, 14, 14] <br>

-> Channel reduction using 1x1 conv: each feature map: [B, 512, 14, 14] -> [B, 32, 14, 14]

-> Upsampling: each feature map: [B, 32, 14, 14] -> [B, 32, 224, 224]

-> Concatenation: 5 maps × 32 channels = [B, 160, 224, 224]

-> Readout network:
      [B, 160, 224, 224] <br>
      -> [B, 16, 224, 224] <br>
      -> [B, 32, 224, 224] <br>
      -> [B, 2, 224, 224] <br>
      -> [B, 1, 224, 224] <br>

-> Raw fixation logits: [B, 1, 224, 224] -> Gaussian smoothing: [B, 1, 224, 224]  -> Add log center bias: [B, 1, 224, 224] -> Final fixation logits: [B, 1, 224, 224]


^^ Loss for this is still BCE, not DeepGaze II actual likelihood loss
