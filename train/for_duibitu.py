import torch
import numpy as np
import open3d as o3d
from dataset import SharpNetCompletionDataset
from model import CompletionTransformer
from loss import CDList,  DensityLoss

batch_size = 3
epoch = 100
lr = 0.0003
k = 20
w = 22
cls_ = 4
param_load_path = "../params/completion-step2-w%d-ptconv-k20-cd-adam-upatte.pth" % w
param_save_path = "../params/completion-step2-w%d-ptconv-k20-cd-adam-upatte.pth" % w

#
# param_load_path = "../params/completion-step2-w%d-ptconv-k20-cd-adam-upatte_.pth" % w
# param_save_path = "../params/completion-step2-w%d-ptconv-k20-cd-adam-upatte_.pth" % w


device = torch.device("cuda:0")
net = CompletionTransformer(n_shift_points=7, d=128, n_encoder=6, n_head=8, k=k, downsample="PointConv")
net.to(device)
net.load_state_dict(torch.load(param_load_path))
loss_fn = CDList()
#loss_fn = EMDList()
density_loss = DensityLoss()
optimizer = torch.optim.Adam(lr=lr, params=net.parameters(), weight_decay=0)

duibitu_dataset = SharpNetCompletionDataset(root='C:/Users/sdnyz/PycharmProjects/dataset/shapenetcore_partanno_segmentation_benchmark_v0/',json_path="train_test_split/duibitu_for_CRC-SSN.json", cls_=cls_, w=w)



def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes)[y.cpu().data.numpy(), ].to(device)



if __name__ == '__main__':
    net.eval()
    with torch.no_grad():
        for i in range(len(duibitu_dataset)):
            remain_pc_list_, crop_list_, _, crop_grid_list_, _, _, cls_ = duibitu_dataset[i]
            for j, remain_pc in enumerate(remain_pc_list_):
                remain_pc = remain_pc.to(device)
                crop_grid = crop_grid_list_[j].to(device)
                inp = torch.cat([remain_pc, crop_grid], dim=0).unsqueeze(0)
                need_num = [crop_grid_list_[j].shape[0]]
                shifted = net(inp, to_categorical(torch.LongTensor([cls_[j]]), 16), need_num)[0].cpu().numpy()
                # open3d
                remain_pts = remain_pc.cpu().numpy()
                remain_pc = o3d.geometry.PointCloud()
                remain_pc.points = o3d.Vector3dVector(remain_pts)
                remain_pc.colors = o3d.Vector3dVector(np.array([[1, 0.706, 0]] * remain_pts.shape[0]))
                # before completion
                crop_pts = crop_grid.cpu().numpy()
                crop_pc = o3d.geometry.PointCloud()
                crop_pc.points = o3d.Vector3dVector(crop_pts)
                crop_pc.colors = o3d.Vector3dVector(np.array([[0, 0.651, 0.929]] * crop_pts.shape[0]))
                # after completion
                shifted_pts = shifted
                shifted_pc = o3d.geometry.PointCloud()
                shifted_pc.points = o3d.Vector3dVector(shifted_pts)
                shifted_pc.colors = o3d.Vector3dVector(np.array([[0, 0.651, 0.929]] * shifted_pts.shape[0]))
                o3d.draw_geometries([remain_pc, crop_pc], window_name="test", width=800, height=600)
                o3d.draw_geometries([remain_pc, shifted_pc], window_name="test", width=800, height=600)
                #
                crop_list__ = np.array(crop_list_[j])
                remain_pc_list__ = np.array(remain_pc_list_[j])
                np.savetxt('../duibitu/I_txt' + str(i) + '_' + str(j) + '.txt', remain_pc_list__, fmt="%f;%f;%f")
                np.savetxt('../duibitu/O_txt' + str(i) + '_' + str(j) + '.txt', shifted, fmt="%f;%f;%f")
                np.savetxt('../duibitu/G_txt' + str(i) + '_' + str(j) + '.txt', crop_list__, fmt="%f;%f;%f")
                #