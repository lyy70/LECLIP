import logging
import cv2
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
import AnomalyCLIP_lib
import torch
import argparse
from utils import winSplit, calWinAnoMap
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from utils import normalize
from dataset import Dataset
from metrics import image_level_metrics, pixel_level_metrics
import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

device = "cuda:4" if torch.cuda.is_available() else "cpu"

##############################################
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc
def apply_ad_scoremap(image, scoremap, alpha=0.6):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)
def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

# 窗口分割器
winSpliter = winSplit()
##############################################
def test(args):
    ##############################################
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    ##############################################
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, 'log.txt')
    # logger日志模块
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')
    ##############################################
    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
                              "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()
    ####################################################
    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform,
                        dataset_name=args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list
    ####################################################
    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['gt_sp'] = []
    results['pr_sp'] = []
    ####################################################
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    ####################################################
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    model.to(device)
    ####################################################
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        win_image = image.clone()
        with torch.no_grad():
            win_images, win_masklist = winSpliter(win_image)
            win_images = F.interpolate(win_images, size=img_size, mode='bilinear', align_corners=True)#[289(winnum * B),3,518,518]
        cls_name = items['cls_name']
        results['cls_names'].append(cls_name[0])

        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask)

        results['gt_sp'].append(items['anomaly'].item())
        with torch.no_grad():
            ####################################################
            image_features, patch_tokens = model.encode_image(image, features_list, DPAM_layer=20)
            win_image_features = []
            win_patch_tokens = []
            for win_img in win_images:
                win_image_feature, win_patch_token = model.encode_image(win_img.unsqueeze(0), features_list, DPAM_layer=20)
                win_image_feature = win_image_feature / win_image_feature.norm(dim=-1, keepdim=True)
                win_image_features.append(win_image_feature.squeeze(0))
                win_patch_token = (win_patch_token[0])[:,1:,:]
                win_patch_tokens.append(win_patch_token.squeeze(0))
            #############################################
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            print("src",text_probs)
            #######################
            win_image_features = torch.stack(win_image_features).to(device)  # [9,768]
            win_text_probs = win_image_features @ text_features.permute(0, 2, 1)
            win_text_probs = (win_text_probs / 0.07).softmax(-1)
            win_text_probs = (win_text_probs[:, :, 1]).unsqueeze(0)
            print(win_text_probs)
            ####################################################
            patch_feature = patch_tokens[0]
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
            anomaly_map = patch_feature @ text_features.permute(0, 2, 1)
            anomaly_map = (anomaly_map[:,1:,:] / 0.07).softmax(-1)
            B, L, C = anomaly_map.shape
            H = int(np.sqrt(L))
            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                        size=img_size, mode='bilinear', align_corners=True)
            anomaly_map = (anomaly_map[:, 1, :, :] + 1 - anomaly_map[:, 0, :, :]) / 2
            anomaly_map = torch.from_numpy(gaussian_filter(anomaly_map.detach().cpu(), sigma=args.sigma))
            ################## visualization ############
            # path = items['img_path']
            # cls = path[0].split('/')[-2]
            # filename = path[0].split('/')[-1]
            # a = cv2.imread(path[0])
            # vis = cv2.cvtColor(cv2.resize(a, (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
            # mask = normalize(anomaly_map[0].squeeze(0).numpy())
            # vis = apply_ad_scoremap(vis, mask)
            # vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
            # save_vis = os.path.join(save_path, 'imgs', cls_name[0], cls, f"anomaly_map")
            # if not os.path.exists(save_vis):
            #     os.makedirs(save_vis)
            # cv2.imwrite(os.path.join(save_vis, filename), vis)
            ###############################################
            w_anomaly_map = []
            win_images_np = win_images.cpu().numpy()
            for i, w_patch_token in enumerate(win_patch_tokens):
                w_patch_token = w_patch_token / w_patch_token.norm(dim=-1, keepdim=True)
                wanomaly_map = w_patch_token @ text_features.permute(0, 2, 1)
                wanomaly_map = (wanomaly_map / 0.07).softmax(-1)
                w_anomaly_map.append(wanomaly_map.squeeze(0))
                ################## visualization ############
                # W_B, W_L, W_C = wanomaly_map.shape
                # W_H = int(np.sqrt(W_L))
                # WIN_anomaly_map = F.interpolate(wanomaly_map.permute(0, 2, 1).view(B, 2, W_H, W_H), size=img_size, mode='bilinear', align_corners=True)
                # WIN_anomaly_map = (WIN_anomaly_map[:, 1, :, :] + 1 - WIN_anomaly_map[:, 0, :, :]) / 2
                # WIN_anomaly_map = torch.from_numpy(gaussian_filter(WIN_anomaly_map.detach().cpu(), sigma=args.sigma))
                #
                # min_val = win_images_np[i].min()
                # max_val = win_images_np[i].max()
                # win_image = win_images_np[i].transpose(1, 2, 0)
                # win_image = (win_image - min_val) / (max_val - min_val)
                # win_image = (win_image * 255).astype(np.uint8)
                # win_image = cv2.cvtColor(win_image, cv2.COLOR_RGB2BGR)
                #
                # save_vis = os.path.join(save_path, 'imgs', cls_name[0], cls, f"Win_image{i}")
                # if not os.path.exists(save_vis):
                #     os.makedirs(save_vis)
                # cv2.imwrite(os.path.join(save_vis, filename), win_image)
                #
                # vis = cv2.cvtColor(cv2.resize(win_image, (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
                # mask = normalize(WIN_anomaly_map[0].squeeze(0).numpy())
                # vis = apply_ad_scoremap(vis, mask)
                # vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
                # save_vis = os.path.join(save_path, 'imgs', cls_name[0], cls, f"Win_map{i}")
                # if not os.path.exists(save_vis):
                #     os.makedirs(save_vis)
                # cv2.imwrite(os.path.join(save_vis, filename), vis)
                #####################################################
            # Slide_Window Merge Function
            w_anomaly_map = torch.stack(w_anomaly_map)#Tensor:[9,1369,2]
            w_anomaly_map = calWinAnoMap(w_anomaly_map, win_masklist, 17, 37, device) #17,13  #[1,1,37,37]
            w_anomaly_map = F.interpolate(w_anomaly_map, size=img_size, mode='bilinear', align_corners=True)
            w_anomaly_map = torch.from_numpy(gaussian_filter(w_anomaly_map[0].detach().cpu(), sigma=args.sigma))#[1,518,518]
            ############# visualization #############
            # path = items['img_path']
            # cls = path[0].split('/')[-2]
            # filename = path[0].split('/')[-1]
            # vis = cv2.cvtColor(cv2.resize(cv2.imread(path[0]), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
            # mask = normalize(w_anomaly_map[0].squeeze(0).numpy())
            # vis = apply_ad_scoremap(vis, mask)
            # vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
            # save_vis = os.path.join(save_path, 'imgs', cls_name[0], cls, f"win_anomaly_map")
            # if not os.path.exists(save_vis):
            #     os.makedirs(save_vis)
            # cv2.imwrite(os.path.join(save_vis, filename), vis)
            ##########################################
            win_avg_anomaly_map = (w_anomaly_map + anomaly_map)/2#
            max_win_anomaly_map = torch.max(w_anomaly_map)
            max_win_anomaly_map = max_win_anomaly_map.unsqueeze(0).to(device)
            win_avg_text_probs = (text_probs + max_win_anomaly_map) / 2#
            ################# visualization ##################
            # path = items['img_path']
            # cls = path[0].split('/')[-2]
            # filename = path[0].split('/')[-1]
            # vis = cv2.cvtColor(cv2.resize(cv2.imread(path[0]), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
            # mask = normalize(win_avg_anomaly_map[0].squeeze(0).numpy())
            # vis = apply_ad_scoremap(vis, mask)
            # vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
            # save_vis = os.path.join(save_path, 'imgs', cls_name[0], cls)
            # if not os.path.exists(save_vis):
            #     os.makedirs(save_vis)
            # cv2.imwrite(os.path.join(save_vis, filename), vis)
            ######################################################
            results['pr_sp'].extend(win_avg_text_probs.detach().cpu())
            results['anomaly_maps'].append(win_avg_anomaly_map.detach().cpu().numpy())

    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    f1_sp_ls = []
    f1_px_ls = []
    aupro_ls = []
    ap_sp_ls = []
    ap_px_ls = []
    for obj in obj_list:
        table = []
        gt_px = []
        pr_px = []
        gt_sp = []
        pr_sp = []
        pr_sp_tmp = []
        table.append(obj)
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes].squeeze(1).numpy())
                pr_px.append(results['anomaly_maps'][idxes])
                pr_sp_tmp.append(np.max(results['anomaly_maps'][idxes]))
                gt_sp.append(results['gt_sp'][idxes])
                pr_sp.append(results['pr_sp'][idxes])
        gt_px = np.array(gt_px)
        gt_sp = np.array(gt_sp)
        pr_px = np.array(pr_px)
        pr_sp = np.array(pr_sp)

        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        auroc_sp = roc_auc_score(gt_sp, pr_sp)
        ap_sp = average_precision_score(gt_sp, pr_sp)
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())

        # f1_sp
        precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
        # f1_px
        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        # aupro
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        aupro = cal_pro_score(gt_px, pr_px)

        table.append(str(np.round(auroc_px * 100, decimals=1)))
        table.append(str(np.round(f1_px * 100, decimals=1)))
        table.append(str(np.round(ap_px * 100, decimals=1)))
        table.append(str(np.round(aupro * 100, decimals=1)))
        table.append(str(np.round(auroc_sp * 100, decimals=1)))
        table.append(str(np.round(f1_sp * 100, decimals=1)))
        table.append(str(np.round(ap_sp * 100, decimals=1)))

        table_ls.append(table)
        auroc_sp_ls.append(auroc_sp)
        auroc_px_ls.append(auroc_px)
        f1_sp_ls.append(f1_sp)
        f1_px_ls.append(f1_px)
        aupro_ls.append(aupro)
        ap_sp_ls.append(ap_sp)
        ap_px_ls.append(ap_px)

    table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(f1_px_ls) * 100, decimals=1)), str(np.round(np.mean(ap_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(aupro_ls) * 100, decimals=1)), str(np.round(np.mean(auroc_sp_ls) * 100, decimals=1)),
                     str(np.round(np.mean(f1_sp_ls) * 100, decimals=1)), str(np.round(np.mean(ap_sp_ls) * 100, decimals=1))])
    results = tabulate(table_ls, headers=['objects', 'auroc_px', 'f1_px', 'ap_px', 'aupro', 'auroc_sp',
                                          'f1_sp', 'ap_sp'], tablefmt="pipe")

    #Image-level
    # table_ls.append(['mean', str(np.round(np.mean(auroc_sp_ls) * 100, decimals=1)),
    #                  str(np.round(np.mean(ap_sp_ls) * 100, decimals=1))])
    # results = tabulate(table_ls, headers=['objects','auroc_sp','ap_sp'], tablefmt="pipe")
    #Pixel-level
    # table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
    #                  str(np.round(np.mean(aupro_ls) * 100, decimals=1))])
    # results = tabulate(table_ls, headers=['objects','auroc_sp','ap_sp'], tablefmt="pipe")
    logger.info("\n%s", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("WinVL", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str,
                        default="/data/LiuYuyao/Dataset/industry_anomaly_detection/mvtec_test/",
                        help="path to test dataset")
    #industry dataset
    #/data/LiuYuyao/Dataset/industry_anomaly_detection/mvtec/
    #/data/LiuYuyao/Dataset/industry_anomaly_detection/visa/
    #/data/LiuYuyao/Dataset/MPDD/
    #/data/LiuYuyao/Dataset/BTAD/
    #/data/LiuYuyao/Dataset/SDD/
    #/data/LiuYuyao/Dataset/DAGM/
    #/data/LiuYuyao/Dataset/DTD_anomaly_detection/

    #compare CVPR2023 medical dataset
    #/data/LiuYuyao/Dataset/medical_anomaly_detection/RESCAD/
    #/data/LiuYuyao/Dataset/medical_anomaly_detection/LiverAD/
    #/data/LiuYuyao/Dataset/medical_anomaly_detection/BrainAD/

    #image-Level dataset
    #/data/LiuYuyao/Dataset/medical_anomaly_detection/Head_CT/
    #/data/LiuYuyao/Dataset/medical_anomaly_detection/BrainMRI/
    #/data/LiuYuyao/Dataset/medical_anomaly_detection/Br35H/
    #/data/LiuYuyao/Dataset/medical_anomaly_detection/COVID-19/
    #/data/LiuYuyao/Dataset/Railway_anomaly_detection/Track_Scene/

    #pixel-Level dataset
    #/data/LiuYuyao/Dataset/medical_anomaly_detection/ISIC/
    #/data/LiuYuyao/Dataset/medical_anomaly_detection/CVC-ColonDB/
    #/data/LiuYuyao/Dataset/medical_anomaly_detection/CVC-ClinicDB/
    #/data/LiuYuyao/Dataset/medical_anomaly_detection/Kvasir/
    #/data/LiuYuyao/Dataset/medical_anomaly_detection/Endo/
    #/data/LiuYuyao/Dataset/medical_anomaly_detection/TN3K/

    parser.add_argument("--save_path", type=str, default='./results/mvtec_test', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/visa_echo/epoch_15.pth',
                        help='path to checkpoint')
    #
    parser.add_argument("--dataset", type=str, default='mvtec_test')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=8, help="zero shot")

    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)