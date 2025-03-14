import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import numpy as np
from collections import OrderedDict

from net_TGDA_7 import TGDA_7
from utils.YUV_RGB import yuv2rgb,yuv_import,rgb_import
import utils
import time
import yaml
import os.path as op
import argparse
from PIL import Image
from tqdm import tqdm


# Checkpoints dir
ckp_path = '/home/zhuqiang/STDF30/exp/TGDA_7_QP37_1_MFQEv2/ckp_240000.pt'
# raw yuv and lq yuv path
# raw_yuv_path = './data/MFQEv2/test_18/raw/RaceHorses_416x240_300.yuv'
# lq_yuv_path = './data//MFQEv2/test_18/HM16.5_LDP/QP37/RaceHorses_416x240_300.yuv'

### Class C
# raw_yuv_path = './data/MFQEv2/test_18/raw/RaceHorses_832x480_300.yuv'    
# lq_yuv_path = './data//MFQEv2/test_18/HM16.5_LDP/QP37/RaceHorses_832x480_300.yuv'

### Class D
# raw_yuv_path = './data/MFQEv2/test_18/raw/BQSquare_416x240_600.yuv'
# lq_yuv_path = './data//MFQEv2/test_18/HM16.5_LDP/QP37/BQSquare_416x240_600.yuv'

### Class E
raw_yuv_path = './data/MFQEv2/test_18/raw/BQTerrace_1920x1080_600.yuv'
lq_yuv_path = './data//MFQEv2/test_18/HM16.5_LDP/QP37/BQTerrace_1920x1080_600.yuv'

vname = lq_yuv_path.split("/")[-1].split('.')[0]
_,wxh,nfs = vname.split('_')
nfs = int(nfs)
w,h = int(wxh.split('x')[0]),int(wxh.split('x')[1])

nfs = min(nfs,200)
save_old = True # False
# need save or not!
save_current = True # False
# this is for our another paper
if 'C2C' in ckp_path:
    m_name = 'C2C'
elif 'TGDA_4'in ckp_path:
    m_name = 'TGDA_4'
else:
    m_name = 'NULL'
outlog='./details/'+m_name+"_"+vname+'.txt'
def receive_arg():
    """Process all hyper-parameters and experiment settings.
    
    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='./config/TGDA/option_TGDA_4_37.yml', 
        help='Path to option YAML file.'
        )
    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log_test.log"
        )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
        )
    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
        )
    opts_dict['test']['checkpoint_save_path'] = (
        f"{opts_dict['train']['checkpoint_save_path_pre']}"
        f"{opts_dict['test']['restore_iter']}"
        '.pt'
        )

    return opts_dict

def f2list_valid(f,nf):
    f2head={
        3:[0,1,2],
        4:[0,2,3],
        5:[0,3,4],
    }
    if(f<3):#list(range(iter_frm - radius, iter_frm + radius + 1))
        return list(range(f-3,f+4))
    elif(f<6):
        head=f2head[f]
    else:
        if (f % 4 == 0):
            head = [f - 8, f - 4, f - 1]
        elif (f % 4 == 1):
            head = [f - 9, f - 5, f - 1]
        elif (f % 4 == 2):
            head = [f - 6, f - 2, f - 1]
        elif (f % 4 == 3):
            head = [f - 7, f - 3, f - 1]
    if (f % 4 == 0):
        tail = [f + 1, f + 4, f + 8]
    elif (f % 4 == 1):
        tail = [f + 1, f + 3, f + 7]
    elif (f % 4 == 2):
        tail = [f + 1, f + 2, f + 6]
    elif (f % 4 == 3):
        tail = [f + 1, f + 5, f + 9]
    if(f>=nf-9):
        tail=set(tail)
        to_del=set([n for n in tail if(n>=nf)])#比nf大的删了
        tail-=to_del
        todo=sorted(list(set(list(range(f+1,f+4)))-tail))[:3-len(tail)]#使用相邻帧补充
        tail=list(tail)+todo
        tail=sorted(list(tail))
    return np.array(head+[f]+tail)

def main():
    # ==========
    # Load pre-trained model
    # ==========
    opts_dict = receive_arg()
    model = TGDA_7(opts_dict=opts_dict['network'])
    msg = f'loading model {ckp_path}...'
    print(msg)
    checkpoint = torch.load(ckp_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])

    msg = f'> model {ckp_path} loaded.'
    msg1 = "Number of Parameters: [{:.1f}]".format(sum([np.prod(p.size()) for p in model.parameters()]))
    print(msg)
    print(msg1)
    model = model.cuda()
    model.eval()

    # ==========
    # Load entire video
    # ==========
    msg = f'loading raw and low-quality yuv...'
    print(msg)
    raw_y ,raw_u ,raw_v = utils.import_yuv(
        seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False
        )
    lq_y ,lq_u ,lq_v = utils.import_yuv(
        seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=False
        )
    raw_y = raw_y.astype(np.float32) / 255.
    lq_y = lq_y.astype(np.float32) / 255.
    msg = '> yuv loaded.'
    print(msg)
    # save
    
    f = open(outlog,"w")
    # ==========
    # Define criterion
    # ==========
    criterion = utils.PSNR()
    unit = 'dB'

    # ==========
    # Test
    # ==========
    ori_psnr_counter = utils.Counter()
    enh_psnr_counter = utils.Counter()
    Sumtime = 0
    for idx in range(nfs):
        # load lq
        # idx_list = list(range(idx-3,idx+4))
        # idx_list = np.clip(idx_list, 0, nfs-1)
        if 'C2C' in ckp_path:
            idx_list = f2list_valid(idx,nfs)
            idx_list = np.clip(idx_list, 0, nfs-1)
        else:
            idx_list = list(range(idx-3,idx+4))
            idx_list = np.clip(idx_list, 0, nfs-1)
            
        input_data = []
        for idx_ in idx_list:
            input_data.append(lq_y[idx_])
        input_data = torch.from_numpy(np.array(input_data))
        input_data = torch.unsqueeze(input_data, 0).cuda()

        # enhance
        with torch.no_grad():
            # if idx == 0:
            strT = time.time()
            enhanced_frm = model(input_data)
            Sumtime += time.time()-strT

    FPS = nfs / Sumtime 
    print('TGDA 7 at Class D', 'FPS:', FPS)

       
    f.close()


if __name__ == '__main__':
    main()