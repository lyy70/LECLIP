################pixel-level##################
python test.py \
--dataset tn3k \
--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/TN3K/ \
--save_path ./results/tn3k \
--checkpoint_path ./checkpoint/mvtec/epoch_15.pth\

python test.py \
--dataset endo \
--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Endo/ \
--save_path ./results/endo \
--checkpoint_path ./checkpoint/mvtec/epoch_15.pth\

python test.py \
--dataset kvasir \
--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Kvasir/ \
--save_path ./results/kvasir \
--checkpoint_path ./checkpoint/mvtec/epoch_15.pth\

python test.py \
--dataset cvc-clinicdb\
--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/CVC-ClinicDB/ \
--save_path ./results/cvc-clinicdb \
--checkpoint_path ./checkpoint/mvtec/epoch_15.pth\

python test.py \
--dataset cvc-colondb \
--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/CVC-ColonDB/ \
--save_path ./results/cvc-colondb \
--checkpoint_path ./checkpoint/mvtec/epoch_15.pth\

python test.py \
--dataset isic \
--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/ISIC/ \
--save_path ./results/isic \
--checkpoint_path ./checkpoint/mvtec/epoch_15.pth\
