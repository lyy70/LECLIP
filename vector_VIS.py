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

device = "cuda:0" if torch.cuda.is_available() else "cpu"

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


##############################################
def test(args):
    ##############################################
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    patchsize = args.patchsize
    winsize = args.winsize
    stride = args.stride

    # 窗口分割器
    winSpliter = winSplit(imgsize=img_size, patchsize=patchsize, winsize=winsize, stride=stride)
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

    save_feat = text_features.squeeze().detach().cpu().numpy()

    np.save('/data/LiuYuyao/Project/WinVL/checkpoint/temp_vectorVIS/epoch15.npy',save_feat)

    print('ALL DONE!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser("WinVL", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str,
                        default="/data/LiuYuyao/Dataset/industry_anomaly_detection/mvtec/",
                        help="path to test dataset")

    parser.add_argument("--save_path", type=str, default='./results/mvtec_14_27_10', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='/data/LiuYuyao/Project/WinVL/checkpoint/mvtec_echo/epoch_15.pth',
                        help='path to checkpoint')
    #
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=8, help="zero shot")
    #window
    parser.add_argument("--winsize", type=int, default=27, help="random seed")
    parser.add_argument("--stride", type=int, default=10, help="zero shot")
    parser.add_argument("--patchsize", type=int, default=14, help="zero shot")

    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)

