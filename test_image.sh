################image-level##################
python test.py \
--dataset headct \
--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Head_CT/ \
--save_path ./results/headct \
--checkpoint_path ./checkpoint/mvtec/epoch_15.pth\

python test.py \
--dataset brainmri \
--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/BrainMRI/ \
--save_path ./results/brainmri \
--checkpoint_path ./checkpoint/mvtec/epoch_15.pth\

python test.py \
--dataset br35H \
--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Br35H/ \
--save_path ./results/br35H \
--checkpoint_path ./checkpoint/mvtec/epoch_15.pth\

python test.py \
--dataset covid-19 \
--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/COVID-19/ \
--save_path ./results/covid-19 \
--checkpoint_path ./checkpoint/mvtec/epoch_15.pth\

