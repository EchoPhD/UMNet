import os
import argparse
import json
import torch
import numpy as np
from torch.nn.functional import threshold, unfold
from dataloaders.HSI_datasets import *
from utils.logger import Logger
import torch.utils.data as data
from utils.helpers import initialize_weights, initialize_weights_new, to_variable, make_patches
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from models.models import MODELS
from utils.metrics import *
import shutil
import torchvision
from torch.distributions.uniform import Uniform
import sys
#import kornia
#from kornia import laplacian, sobel
from scipy.io import savemat
import torch.nn.functional as F
from utils.vgg_perceptual_loss import VGGPerceptualLoss, VGG19
from utils.spatial_loss import Spatial_Loss

''' \\ -2023-0323 add - \\'''
from visdom import Visdom
import numpy as np
import time
from tqdm import tqdm

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

__dataset__ = {"pavia_dataset": pavia_dataset, "chikusei_dataset": chikusei_dataset, "botswana4_dataset": botswana4_dataset}

# 超参数文件路径
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-c', '--config', default='configs/config_PANNET.json', type=str, help='Path to the config file')
parser.add_argument('-r', '--resume', default=None, type=str, help='Path to the .pth model checkpoint to resume training')
parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
parser.add_argument('--local', action='store_true', default=False)
args = parser.parse_args()

# 加载配置文件
config = json.load(open(args.config))
torch.backends.cudnn.benchmark = True

# 设置种子
torch.manual_seed(7)

# 设置可用于训练的gpu数量
num_gpus = torch.cuda.device_count()

# 模型选择(预训练或正式训练)
model = MODELS[config["model"]](config)
# print(f'\n{model}\n')

# 单卡或者多卡训练
if num_gpus > 1:
    print("Training with multiple GPUs ({})".format(num_gpus))
    model = nn.DataParallel(model).cuda()
else:
    print("Single Cuda Node is avaiable")
    model.cuda()

# 建立训练和测试数据存档
print("Training with dataset => {}".format(config["train_dataset"]))
train_loader = data.DataLoader(__dataset__[config["train_dataset"]](config, is_train=True),
                                batch_size=config["train_batch_size"],
                                num_workers=config["num_workers"],
                                shuffle=True,
                                pin_memory=False,)

test_loader = data.DataLoader(__dataset__[config["train_dataset"]](config,is_train=False),
                                batch_size=config["val_batch_size"],
                                num_workers=config["num_workers"],
                                shuffle=True,
                                pin_memory=False,)

# 初始化超参数
start_epoch = 1
total_epochs = config["trainer"]["total_epochs"]

# 设置优化器
if config["optimizer"]["type"] == "SGD":
    optimizer = optim.SGD(  model.parameters(), 
                            lr=config["optimizer"]["args"]["lr"], 
                            momentum = config["optimizer"]["args"]["momentum"], 
                            weight_decay= config["optimizer"]["args"]["weight_decay"])
elif config["optimizer"]["type"] == "ADAM":
    optimizer = optim.Adam( model.parameters(), 
                            lr=config["optimizer"]["args"]["lr"],
                            weight_decay= config["optimizer"]["args"]["weight_decay"])
else:
    exit("Undefined optimizer type")

# 学习率分配器。(step_size: 每训练step_size个epoch，更新一次参数；)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["optimizer"]["step_size"], gamma=config["optimizer"]["gamma"])

# 加载训练好的模型或者预训练模型
if args.resume is not None:
    print("Loading from existing FCN and copying weights to continue....")
    checkpoint = torch.load(args.resume)
    '''
    checkpoint.pop("low_cross_attention.temperature")
    checkpoint.pop("high_hor_cross_attention.temperature")
    checkpoint.pop("high_ver_cross_attention.temperature")
    checkpoint.pop("high_dia_cross_attention.temperature")
    checkpoint.pop("cross_attention_02.temperature")
    checkpoint.pop("cross_attention_01.temperature")
    '''
    model.load_state_dict(checkpoint, strict=False)
else:
    # initialize_weights(model)
    initialize_weights_new(model)

# 建立损失函数
if config[config["train_dataset"]]["loss_type"] == "L1":
    criterion   = torch.nn.L1Loss()
    HF_loss     = torch.nn.L1Loss()
elif config[config["train_dataset"]]["loss_type"] == "MSE":
    criterion   = torch.nn.MSELoss()
    HF_loss     = torch.nn.MSELoss()
else:
    exit("Undefined loss type")

# 训练函数
def train(epoch):
    train_loss = 0.0
    model.train()
    optimizer.zero_grad()
    # for i, data in enumerate(train_loader, 0):
    for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), leave=True):
        # 读取数据
        _, MS_image, PAN_image, reference = data

        # 取model输出…
        MS_image    = Variable(MS_image.float().cuda()) 
        PAN_image   = Variable(PAN_image.float().cuda())
        reference = Variable(reference.float().cuda())
        out         = model(MS_image, PAN_image, reference)
        outputs = out["pred"]
        s_spatial = out["s_spa"]
        s_spectral = out["s_spe"]

        '''计算损失'''
        # 重建损失
        if config[config["train_dataset"]]["Normalized_L1"]:
            max_ref = torch.amax(reference, dim=(2, 3)).unsqueeze(2).unsqueeze(3).expand_as(reference).cuda()
            l1_error = criterion(outputs / max_ref,  reference / max_ref)  # (B, C, H, W)
            total_uncertainty = torch.exp(-s_spatial / max_ref - s_spectral / max_ref)  # (B, C, H, W)
            weighted_l1 = total_uncertainty * l1_error
            reg_term = 2 * (s_spatial / max_ref + s_spectral / max_ref)
            loss = weighted_l1.mean() + reg_term.mean()
        else:
            l1_error = torch.abs(outputs - reference)  # (B, C, H, W)
            # l1_error = criterion(outputs, reference)  # (B, C, H, W)
            total_uncertainty = torch.exp(-s_spatial - s_spectral)  # (B, C, H, W)
            weighted_l1 = total_uncertainty * l1_error
            reg_term = 2 * (s_spatial + s_spectral)
            # print("weighted_l1.mean()", weighted_l1.mean().data)
            # print("reg_term.mean()", reg_term.mean().data)
            # print("l1_error.mean()", l1_error.mean().data)

            loss = weighted_l1.mean() + reg_term.mean()

        # focus loss
        if config[config["train_dataset"]]["F_LOSS"]:
            loss = loss + out["LF"]

        # L_U loss
        if config[config["train_dataset"]]["LU_LOSS"]:
            loss = loss + out["LU"]

        torch.autograd.backward(loss)
        
        # 网络更新
        if i % config["trainer"]["iter_size"] == 0 or i == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

    writer.add_scalar('Loss/train', loss, epoch)
    
# Testing epoch.
def test(epoch):
    test_loss   = 0.0
    lf_loss_sum = 0.0
    lu_loss_sum = 0.0
    cc          = 0.0
    sam         = 0.0
    rmse        = 0.0
    ergas       = 0.0
    psnr        = 0.0
    val_outputs = {}
    model.eval()
    pred_dic = {}
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader), leave=True):
            image_dict, MS_image, PAN_image, reference = data

            MS_image    = Variable(MS_image.float().cuda()) 
            PAN_image   = Variable(PAN_image.float().cuda())
            reference   = Variable(reference.float().cuda())
            # 获取model输出
            out     = model(MS_image, PAN_image, reference)
            outputs = out["pred"]
            s_spatial = out["s_spa"]
            s_spectral = out["s_spe"]

            # 计算验证损失
            # s = torch.exp(-auout)
            # sr_ = torch.mul(outputs, s)
            # hr_ = torch.mul(reference, s)
            # loss = criterion(sr_, hr_) + 2 * torch.mean(auout)

            l1_error = torch.abs(outputs - reference)  # (B, C, H, W)
            # l1_error = criterion(outputs, reference)  # (B, C, H, W)
            total_uncertainty = torch.exp(-s_spatial - s_spectral)  # (B, C, H, W)
            weighted_l1 = total_uncertainty * l1_error
            reg_term = 2 * (s_spatial + s_spectral)
            loss = weighted_l1.mean() + reg_term.mean()


            # focus loss
            if config[config["train_dataset"]]["F_LOSS"]:
                focus_loss = out["LF"]
                # print("focus_loss:-------->", focus_loss)
                loss = loss + focus_loss
                lf_loss_sum += focus_loss.item()

            # L_U loss
            if config[config["train_dataset"]]["LU_LOSS"]:
                lu_loss = out["LU"]
                # print("lu_loss:-------->", lu_loss)
                loss = loss + lu_loss
                lu_loss_sum += lu_loss.item()

            test_loss   += loss.item()

            ''' Scalling 这块是干啥的？'''
            outputs[outputs<0]      = 0.0
            outputs[outputs>1.0]    = 1.0
            outputs                 = torch.round(outputs*config[config["train_dataset"]]["max_value"])
            pred_dic.update({image_dict["imgs"][0].split("/")[-1][:-4]+"_pred": torch.squeeze(outputs).permute(1,2,0).cpu().numpy()})
            reference               = torch.round(reference.detach()*config[config["train_dataset"]]["max_value"])

            '''计算性能指标'''
            cc += cross_correlation(outputs, reference) # Cross-correlation
            sam += SAM(outputs, reference)  # SAM
            rmse += RMSE(outputs/torch.max(reference), reference/torch.max(reference))  # RMSE
            beta = torch.tensor(config[config["train_dataset"]]["HR_size"]/config[config["train_dataset"]]["LR_size"]).cuda()   # ERGAS
            ergas += ERGAS(outputs, reference, beta)
            psnr += PSNR(outputs, reference)    # PSNR

    # 对测试集的性能指标取平均值
    cc /= len(test_loader)
    sam /= len(test_loader)
    rmse /= len(test_loader)
    ergas /= len(test_loader)
    psnr /= len(test_loader)

    #返回指标输出
    metrics = { "loss": float(test_loss),
                #"LF_loss": float(lf_loss_sum),
                "LU_loss": float(lu_loss_sum),
                "cc": float(cc), 
                "sam": float(sam), 
                "rmse": float(rmse), 
                "ergas": float(ergas), 
                "psnr": float(psnr)}
    # print("LOSS: %.6f\tLF_loss: %.6f\tLU_loss: %.6f\tCC: %.6f\tSAM: %.6f\tPSNR:%.6f" %(test_loss, lf_loss_sum, lu_loss_sum, cc, sam, psnr))
    print("LOSS: %.6f\tLU_loss: %.6f\tCC: %.6f\tSAM: %.6f\tPSNR:%.6f" %(test_loss, lu_loss_sum, cc, sam, psnr))
    return image_dict, pred_dic, metrics


'''以下是主程序入口'''
#if __name__ == '__main__':
# 设置tensorboard并复制.json文件保存目录。
PATH = "./"+config["experim_name"]+"/"+config["train_dataset"]+"/"+"N_modules("+str(config["N_modules"])+")"
ensure_dir(PATH+"/")
writer = SummaryWriter(log_dir=PATH)
shutil.copy2(args.config, PATH)

# 打印模型到文本文件
original_stdout = sys.stdout 
with open(PATH+"/"+"model_summary.txt", 'w+') as f:
    sys.stdout = f
    # print(f'\n{model}\n')
    sys.stdout = original_stdout 

# 主循环
viz = Visdom()
viz.line([0.], [0], win='TRAIN_SAM_test_pre', opts=dict(title='TRAIN_SAM_test_pre', legend=['SAM']))
viz.line([0.], [0], win='TRAIN_PSNR_test_pre', opts=dict(title='TRAIN_PSNR_test_pre', legend=['PSNR']))
#best_sam = 100.0
best_psnr = 0.0
#best_loss = 1000000
for epoch in range(start_epoch, total_epochs):
    scheduler.step(epoch)
    print("\nTraining Epoch: %d" % epoch)
    #训练
    train(epoch)

    if epoch % config["trainer"]["test_freq"] == 0:
        print("\nTesting Epoch: %d" % epoch)
        #测试
        image_dict, pred_dic, metrics=test(epoch)
        plt_PSNR = metrics["psnr"]
        viz.line([plt_PSNR], [epoch], win='TRAIN_PSNR_test_pre', update='append')
        time.sleep(0.5)
        
        plt_SAM = metrics["sam"]
        viz.line([plt_SAM], [epoch], win='TRAIN_SAM_test_pre', update='append')
        time.sleep(0.5)
                
        #保存最好的模型
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            #保存最佳性能指标s
            torch.save(model.state_dict(), PATH+"/"+"best_model.pth")
            with open(PATH+"/"+" bast_metrics.json", "w+") as outfile:
                json.dump(metrics, outfile)
            #保存最好的预测
            savemat(PATH+"/"+ "final_prediction.mat", pred_dic)
    
