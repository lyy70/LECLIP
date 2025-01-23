################image_pixel-level##################
python test.py \
--dataset mvtec \
--data_path /data/LiuYuyao/Dataset/MVTec_anomaly_detection/mvtec_anomaly_detection/ \
--save_path ./results/mvtec \
--checkpoint_path ./checkpoint/visa/epoch_15.pth \

python test.py \
--dataset visa \
--data_path /data/LiuYuyao/Dataset/Visa_anomaly_detection/ \
--save_path ./results/visa \
--checkpoint_path ./checkpoint/mvtec/epoch_15.pth \

python test.py \
--dataset mpdd \
--data_path /data/LiuYuyao/Dataset/MPDD/ \
--save_path ./results/mpdd \
--checkpoint_path ./checkpoint/mvtec/epoch_15.pth\

python test.py \
--dataset btad \
--data_path /data/LiuYuyao/Dataset/BTAD/ \
--save_path ./results/btad \
--checkpoint_path  ./checkpoint/mvtec/epoch_15.pth\

python test.py \
--dataset sdd \
--data_path /data/LiuYuyao/Dataset/SDD/ \
--save_path ./results/sdd \
--checkpoint_path ./checkpoint/mvtec/epoch_15.pth\

python test.py \
--dataset dagm \
--data_path /data/LiuYuyao/Dataset/DAGM/ \
--save_path ./results/dagm \
--checkpoint_path ./checkpoint/mvtec/epoch_15.pth\

python test.py \
--dataset dtd \
--data_path /data/LiuYuyao/Dataset/DTD_anomaly_detection/ \
--save_path ./results/dtd \
--checkpoint_path ./checkpoint/mvtec/epoch_15.pth\