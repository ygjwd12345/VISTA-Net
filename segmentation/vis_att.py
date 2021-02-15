import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import numpy as np
img=cv2.imread('./att_seg_rgb.jpg')
# img=img.transpose((1,2,0))
print(img.shape)
att_dict=sio.loadmat('/data2/gyang/PGA-net/segmentation/sp_ag_weights_2020-06-05-16-13-46.mat')
print(att_dict['sp_feat'].shape)
atts=np.squeeze(att_dict['sp_feat'])
num=atts.shape
for i in range(num[0]):
        att=atts[i]
        # print(att)
        att=cv2.resize(att,(500,375))
        jet = plt.get_cmap('jet')
        cNorm = colors.Normalize()
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        colorVal = scalarMap.to_rgba(255-att)
        filename='./att/sp_'+str(i).zfill(3)+'.png'
        plt.imsave(filename, colorVal)

# att_dict=sio.loadmat('/data2/gyang/PGA-net/segmentation/att2020-06-05-15-51-20.mat')
# atts = np.squeeze(att_dict['att'])
# print(att_dict['att'].shape)
#
# num=atts.shape
# for i in range(num[0]):
#     for j in range(num[1]):
#         att=atts[i,j]
#         # print(att)
#         att=cv2.resize(att,(375,500))
#         jet = plt.get_cmap('jet')
#         cNorm = colors.Normalize()
#         scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#         colorVal = scalarMap.to_rgba(att)
#         filename='./att/att_'+str(i)+'_'+str(j).zfill(3)+'.png'
#         plt.imsave(filename, colorVal)


