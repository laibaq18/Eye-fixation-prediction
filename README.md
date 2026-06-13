Go to the folder where scripts and data is:

1) Create virtual environment and install dependencies
  
2)
Train via: 
``python3 train.py --data-root ../cv2_project_data --epochs 5 --batch-size 4 --freeze-backbone``

3)
Create predictions via:
``python predict_test.py   --data-root ../cv2_project_data   --checkpoint checkpoints/best.pt   --save-dir test_predictions``
