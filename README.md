#TODO: ROC-AUC Comparison of all models

Go to the folder where scripts and data is:

1) Create virtual environment and install dependencies
  
2) Train via: 
``python3 train.py --data-root ../cv2_project_data --model_name FCN-resnet50 / deepgaze2 / SAM``

3) Create predictions via:
``python3 predict_test.py   --data-root ../cv2_project_data  --model_name FCN-resnet50 / deepgaze2 / SAM   --save-dir test_predictions``

FCN Resnet50:
Input image : [B, 3, 224, 224] -> FCN ResNet-50 -> Raw logits : [B, 1, 224, 224] -> Gaussian smoothing : [B, 1, 224, 224] -> Add log center bias : [B, 1, 224, 224] -> Final fixation logits : [B, 1, 224, 224]

^ uses BCE loss

<br>
<br>

DeepGaze2 Like:
Input image : [B, 3, 224, 224] -> Frozen VGG-19 feature extractor -> Selected VGG features:
      conv5_1 : [B, 512, 14, 14] <br>
      relu5_1 : [B, 512, 14, 14] <br>
      relu5_2 : [B, 512, 14, 14] <br>
      conv5_3 : [B, 512, 14, 14] <br>
      relu5_4 : [B, 512, 14, 14] <br>


-> Upsampling: each feature map: [B, 512, 14, 14] -> [B, 512, 112, 112]

-> Concatenation: 5 maps × 512 channels = [B, 2560, 112, 112]

-> Readout network:
      [B, 2560, 112, 112] <br>
      -> [B, 16, 112, 112] <br>
      -> [B, 32, 112, 112] <br>
      -> [B, 2, 112, 112] <br>
      -> [B, 1, 112, 112] <br>

-> Raw fixation logits: [B, 1, 112, 112] -> Upsample to orig img size: [B, 1, 224, 224] -> Gaussian smoothing: [B, 1, 224, 224]  -> Add log center bias: [B, 1, 224, 224] -> Final fixation logits: [B, 1, 224, 224]


^^ uses DeepGaze II likelihood loss
