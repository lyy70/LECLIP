#################pixel-level##################
#python test.py \
#--dataset tn3k \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/TN3K/ \
#--save_path ./results/tn3k14_31_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth\
#--winsize 31 \
#--stride  6

#python test.py \
#--dataset endo \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Endo/ \
#--save_path ./results/endo_14_27_10 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 27 \
#--stride  10
#
#python test.py \
#--dataset endo \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Endo/ \
#--save_path ./results/endo_14_31_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 31 \
#--stride  6
#
#python test.py \
#--dataset kvasir \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Kvasir/ \
#--save_path ./results/kvasir_14_27_10 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 27 \
#--stride  10
#
#python test.py \
#--dataset kvasir \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Kvasir/ \
#--save_path ./results/kvasir_14_31_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 31 \
#--stride  6

#python test.py \
#--dataset cvc-clinicdb \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/CVC-ClinicDB/ \
#--save_path ./results/cvc-clinicdb_14_27_10 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 27 \
#--stride  10

#python test.py \
#--dataset cvc-clinicdb \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/CVC-ClinicDB/ \
#--save_path ./results/cvc-clinicdb_14_31_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 31 \
#--stride  6

#python test.py \
#--dataset cvc-colondb \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/CVC-ColonDB/ \
#--save_path ./results/cvc-colondb_14_27_10 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 27 \
#--stride  10
#
#python test.py \
#--dataset cvc-colondb \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/CVC-ColonDB/ \
#--save_path ./results/cvc-colondb_14_31_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 31 \
#--stride  6

#python test.py \
#--dataset isic \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/ISIC/ \
#--save_path ./results/isic \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth\
#--winsize 31 \
#--stride  6
#################image-level##################
#python test.py \
#--dataset headct \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Head_CT/ \
#--save_path ./results/headct \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth\
#
#python test.py \
#--dataset brainmri \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/BrainMRI/ \
#--save_path ./results/brainmri \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth\

#python test.py \
#--dataset brainmri \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/BrainMRI/ \
#--save_path ./results/brainmri_14_31_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 31 \
#--stride  6

#python test.py \
#--dataset br35H \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Br35H/ \
#--save_path ./results/br35H \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth\

#python test.py \
#--dataset br35H \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Br35H/ \
#--save_path ./results/br35H_14_31_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 31 \
#--stride  6
#
#python test.py \
#--dataset covid-19 \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/COVID-19/ \
#--save_path ./results/covid-19 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth\#

#python test.py \
#--dataset covid-19 \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/COVID-19/ \
#--save_path ./results/covid-19_14_31_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 31 \
#--stride  6
#################image_pixel-level##################
python test.py \
--dataset mvtec \
--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/mvtec/ \
--save_path ./results/mvtec_qkv \
--checkpoint_path ./checkpoint/visa_qkv/epoch_15.pth \
--winsize 27 \
--stride 10

python test.py \
--dataset visa \
--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/visa/ \
--save_path ./results/visa_qkv \
--checkpoint_path ./checkpoint/mvtec_qkv/epoch_15.pth \
--winsize 27 \
--stride 10

#python test.py \
#--dataset mpdd \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/MPDD/ \
#--save_path ./results/mpdd_14_27_10 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 27 \
#--stride 10
#
#python test.py \
#--dataset btad \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/BTAD/ \
#--save_path ./results/btad_14_27_10 \
#--checkpoint_path  ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 27 \
#--stride 10
#
#python test.py \
#--dataset sdd \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/SDD \
#--save_path ./results/sdd_14_27_10 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 27 \
#--stride 10
#
#python test.py \
#--dataset dagm \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/DAGM \
#--save_path ./results/dagm_14_27_10 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 27 \
#--stride 10
#
#python test.py \
#--dataset dtd \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/DTD/ \
#--save_path ./results/dtd_14_27_10 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 27 \
#--stride 10
########################################################
#python test.py \
#--dataset mvtec \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/mvtec/ \
#--save_path ./results/mvtec_0.4_27_10 \
#--checkpoint_path ./checkpoint/visa_0.4/epoch_15.pth \
#--winsize 27 \
#--stride  10
#
#python test.py \
#--dataset mvtec \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/mvtec/ \
#--save_path ./results/mvtec_0.6_27_10 \
#--checkpoint_path ./checkpoint/visa_0.6/epoch_15.pth \
#--winsize 27 \
#--stride  10
#
#python test.py \
#--dataset mvtec \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/mvtec/ \
#--save_path ./results/mvtec_0.8_27_10 \
#--checkpoint_path ./checkpoint/visa_0.8/epoch_15.pth \
#--winsize 27 \
#--stride  10
#
#python test.py \
#--dataset mvtec \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/mvtec/ \
#--save_path ./results/mvtec_0.9_27_10 \
#--checkpoint_path ./checkpoint/visa_0.9/epoch_15.pth \
#--winsize 27 \
#--stride  10
#
#python test.py \
#--dataset visa \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/visa/ \
#--save_path ./results/visa_0.4_27_10 \
#--checkpoint_path ./checkpoint/mvtec_0.4/epoch_15.pth \
#--winsize 27 \
#--stride  10
#
#python test.py \
#--dataset visa \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/visa/ \
#--save_path ./results/visa_0.6_27_10 \
#--checkpoint_path ./checkpoint/mvtec_0.6/epoch_15.pth \
#--winsize 27 \
#--stride  10
#
#python test.py \
#--dataset visa \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/visa/ \
#--save_path ./results/visa_0.8_27_10 \
#--checkpoint_path ./checkpoint/mvtec_0.8/epoch_15.pth \
#--winsize 27 \
#--stride  10
#
#python test.py \
#--dataset visa \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/visa/ \
#--save_path ./results/visa_0.9_27_10 \
#--checkpoint_path ./checkpoint/mvtec_0.9/epoch_15.pth \
#--winsize 27 \
#--stride  10
####################################################################
#python test.py \
#--dataset mvtec \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/mvtec/ \
#--save_path ./results/mvtec4_31_6 \
#--checkpoint_path ./checkpoint/visa_echo/epoch_15.pth \
#--winsize 31 \
#--stride  6

#python test.py \
#--dataset mvtec \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/mvtec/ \
#--save_path ./results/mvtec14_27_10 \
#--checkpoint_path ./checkpoint/visa_echo/epoch_15.pth \
#--winsize 27 \
#--stride  10

#python test.py \
#--dataset mvtec \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/mvtec/ \
#--save_path ./results/mvtec14_25_12 \
#--checkpoint_path ./checkpoint/visa_echo/epoch_15.pth \
#--winsize 25 \
#--stride  12

#python test.py \
#--dataset mvtec \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/mvtec/ \
#--save_path ./results/mvtec14_25_6 \
#--checkpoint_path ./checkpoint/visa_echo/epoch_15.pth \
#--winsize 25 \
#--stride  6

#python test.py \
#--dataset mvtec \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/mvtec/ \
#--save_path ./results/mvtec14_13_12 \
#--checkpoint_path ./checkpoint/visa_echo/epoch_15.pth \
#--winsize 13 \
#--stride 12

#python test.py \
#--dataset mvtec \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/mvtec/ \
#--save_path ./results/mvtec14_19_6 \
#--checkpoint_path ./checkpoint/visa_echo/epoch_15.pth \
#--winsize 19 \
#--stride  6

#python test.py \
#--dataset visa \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/visa/ \
#--save_path ./results/visa14_31_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 31 \
#--stride  6
#
#python test.py \
#--dataset visa \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/visa/ \
#--save_path ./results/visa14_27_10 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 27 \
#--stride  10
#
#python test.py \
#--dataset visa \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/visa/ \
#--save_path ./results/visa14_25_12 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 25 \
#--stride  12
#
#python test.py \
#--dataset visa \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/visa/ \
#--save_path ./results/visa14_25_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 25 \
#--stride  6

#python test.py \
#--dataset visa \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/visa/ \
#--save_path ./results/visa14_13_12 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 13 \
#--stride  12
#
#python test.py \
#--dataset visa \
#--data_path /data/LiuYuyao/Dataset/industry_anomaly_detection/visa/ \
#--save_path ./results/visa14_19_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 19 \
#--stride  6
#
#python test.py \
#--dataset headct \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Head_CT/ \
#--save_path ./results/headct14_31_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 31 \
#--stride  6
#
#python test.py \
#--dataset headct \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Head_CT/ \
#--save_path ./results/headct14_27_10 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 27 \
#--stride  10

#python test.py \
#--dataset headct \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Head_CT/ \
#--save_path ./results/headct14_17_10 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 17 \
#--stride  10

#python test.py \
#--dataset headct \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Head_CT/ \
#--save_path ./results/headct14_25_12 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 25 \
#--stride  12
#
#python test.py \
#--dataset headct \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Head_CT/ \
#--save_path ./results/headct14_25_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 25 \
#--stride  6
#
#python test.py \
#--dataset headct \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Head_CT/ \
#--save_path ./results/headct14_13_12 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 13 \
#--stride 12
#
#python test.py \
#--dataset headct \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/Head_CT/ \
#--save_path ./results/headct14_19_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 19 \
#--stride  6

#python test.py \
#--dataset isic \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/ISIC/ \
#--save_path ./results/isic14_31_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 31 \
#--stride  6
#
#python test.py \
#--dataset isic \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/ISIC/ \
#--save_path ./results/isic14_27_10 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 27 \
#--stride  10
#
#python test.py \
#--dataset isic \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/ISIC/ \
#--save_path ./results/isic14_25_12 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 25 \
#--stride  12
#
#python test.py \
#--dataset isic \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/ISIC/ \
#--save_path ./results/isic14_25_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 25 \
#--stride  6
#
#python test.py \
#--dataset isic \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/ISIC/ \
#--save_path ./results/isic14_13_12 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 13 \
#--stride  12
#
#python test.py \
#--dataset isic \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/ISIC/ \
#--save_path ./results/isic14_19_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 19 \
#--stride  6
#
#python test.py \
#--dataset tn3k \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/TN3K/ \
#--save_path ./results/tn3k14_31_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 31 \
#--stride  6
#
#python test.py \
#--dataset tn3k \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/TN3K/ \
#--save_path ./results/tn3k14_27_10 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 27 \
#--stride  10
#
#python test.py \
#--dataset tn3k \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/TN3K/ \
#--save_path ./results/tn3k14_25_12 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 25 \
#--stride  12
#
#python test.py \
#--dataset tn3k \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/TN3K/ \
#--save_path ./results/tn3k14_25_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 25 \
#--stride  6
#
#python test.py \
#--dataset tn3k \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/TN3K/ \
#--save_path ./results/tn3k14_13_12 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 13 \
#--stride  12
#
#python test.py \
#--dataset tn3k \
#--data_path /data/LiuYuyao/Dataset/medical_anomaly_detection/TN3K/ \
#--save_path ./results/tn3k14_19_6 \
#--checkpoint_path ./checkpoint/mvtec_echo/epoch_15.pth \
#--winsize 19 \
#--stride  6
