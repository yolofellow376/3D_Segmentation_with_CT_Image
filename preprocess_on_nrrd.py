#!/usr/bin/env python
# coding: utf-8

# In[17]:


import open3d as o3d
import nrrd
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from skimage.transform import rescale, resize, downscale_local_mean
from pydicom.data import get_testdata_files
from collections import Counter
from scipy.ndimage import zoom
pat_dir='C:/Users/56-000M100-32/Arthur/datasets_graz/graz/Mandibular_Datasets/'
subjects= os.listdir(pat_dir)


# In[18]:


def o3d_visualize(pcs,colors): #visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pcs))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[64, 64, 64],
                                  up=[-0.0694, -0.9768, 0.2024])


# In[19]:


def o3d_save(pcs,colors,path): #save a scene
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pcs))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    o3d.io.write_point_cloud(path, pcd)


# In[20]:


def o3d_load(path): #load a scene
    pcd_load = o3d.io.read_point_cloud(path)
    return pcd_load


# In[21]:


def v_to_pc(arr,z,count,pcs,colors,threshold): #voxel to pointcloud
    for y,row in enumerate(arr):
        for x,col in enumerate(row):
            if col >=threshold :
                pcs.append([x,y,z])
                colors.append(col)


# In[22]:


def get_threshold(old_threshold,old_info,new_info):
    
    #_info = [max,mean,min,std]
    factor=(old_threshold-old_info[1])/old_info[3]
    new_threshold=new_info[1]+new_info[3]*factor  
    
    return new_threshold


# In[23]:


def get_pcs(new_threshold,cube): #generate validated point clouds and colors
    count=0
    pcs=[]
    colors=[]
    for z,ds in enumerate(cube):
        v_to_pc(ds,z,count,pcs,colors,new_threshold)
        #print(pcs[-100:])

    return pcs,colors


# In[24]:


def get_3d(sub): #get whole cube 3d array 
    filename=pat_dir+sub
    readdata, header = nrrd.read(filename)
    print(readdata.shape)
    readdata = np.einsum('ijk->kji', readdata) #x,y,z -> z,y,x
    print(readdata.shape)
    return readdata


# In[25]:


def filter_eyebox(cube): #filter out eyebox, maybe just filter out 1024?
    colors= cube.flatten()
    eyebox_color = Counter(colors).most_common()
    for i in eyebox_color:
        if i[0]>0:
            print(i)
            break
    target = i[0]
    for i,img in enumerate(cube):
        for j,row in enumerate(img):
            for k,c in enumerate(row):
                if c==target:
                    cube[i][j][k]=-1000
    return cube


# In[26]:


def fn(sub,input_size,old_threshold): #generate pointclouds and colors
    
    #step1: get 3d array
    cube = get_3d(sub)   #z,y,x
    
    plt.imshow(cube[0])  #yaxis=y,xaxis=x
    plt.scatter(x=0,y=128,color='r',s=1) 
    plt.show()
    #step2: filter out eyebox
    cube = filter_eyebox(cube) #z,y,x
    
    #step3: resize
    ratio = (input_size/cube.shape[0],input_size/cube.shape[1],input_size/cube.shape[2])
    old_info = [cube.max(),cube.mean(),cube.min(),cube.std()]
    print('old: ', old_info)

    cube = zoom(cube, ratio)  #z,y,x
    new_info = [cube.max(),cube.mean(),cube.min(),cube.std()]
    print('new: ', new_info)
    print(cube.shape)
    
    #step3: derive threshold for bone
    new_threshold  = get_threshold(old_threshold,old_info,new_info)
    print(new_threshold, old_threshold )  
    
    #step4: get all points whose greyscale are greater than new_threshold
    pcs, colors = get_pcs(new_threshold,cube) #voxel(z,y,x) -> pcs(x,y,z)
   
    
    
    return np.array(pcs),np.array(colors)


# In[27]:


def clustering(pcs,colors,MAX_DISTANCE=25,QUEUE_MAX_LENGTH=300): #self-made over-segmentation, further add colors in formula
    seg=[]
    count=0
    seg_idx=0
    anchors=[]
    anchor_x=-100
    anchor_y=-100
    anchor_z=-100
    
    
    for index,pc in enumerate(pcs):
        
        #initial
        x,y,z=pc
        target_anchor=None
        near_an_anchor=False
        min_distance = 100
        
        #search closet anchor point
        for pop in anchors:
            p1=np.array(pc)
            p2=np.array(pop[:-1])
            squared_dist = np.sum((p1-p2)**2, axis=0)# + abs(colors[index][0]-colors[pop[3]][0])*60**2
            #print(squared_dist, abs(colors[index][0]-colors[pop[3]][0])*60**2)
            if squared_dist<=MAX_DISTANCE and  squared_dist < min_distance :
                near_an_anchor=True
                target_anchor=pop[3]
                min_distance = squared_dist
        
        #update seg[] or seg[] & anchor[]
        if  near_an_anchor:
            #print('same')
            seg.append(target_anchor)
        else:
            #print('add new anchor')
            if len(anchors)>QUEUE_MAX_LENGTH:
                _null=anchors.pop(0)
            anchor_x=x
            anchor_y=y
            anchor_z=z
            seg_idx=index
            seg.append(index)
            anchors.append([anchor_x,anchor_y,anchor_z,seg_idx])
            count=count+1
        
        #print(z,target_anchor,len(anchors),end=" ")
        


    print('cluster count: ',count)
    print(len(seg))
    return seg


# In[28]:


for sub in subjects:
#sub = subjects[3]
    #step1: preprocess pointclouds and colors
    pcs,colors = fn(sub,input_size=128,old_threshold = 400)
    
    #step2: linear scale colors to 0~1 and save in .ply
    '''
    colors=preprocessing.minmax_scale(colors, feature_range=(0,1))
    tmp=[]
    for i in colors:
        tmp.append([i,i,i])
 
    colors=np.array(tmp)
    
    #o3d_save(pcs,colors,'./temp_thd400_'+sub+'.ply')
    '''
    print(pcs.shape)
    
    #step3: clustering
    seg =clustering(pcs,colors)
    
    #step4: linear scale seg to 0~1 and save in .ply
    c=preprocessing.minmax_scale(seg, feature_range=(0,1))
    tmp=[]
    for i in c:
        tmp.append([i,i,i])  


    writepath = './0825_sz128_dist25_thd400_'+sub[:-5]
    o3d_save(pcs,tmp,writepath+'.ply') 
    
    #step4.5: make sure data in correct form
    seg =np.array(seg)
    colors =np.array(colors)
    pcs =np.array(pcs)
    
    #step5: store data in txt
    txt_path=writepath+'.txt'
    f =open(txt_path,'w')
    for i,line in enumerate(pcs):
        f.write(str(line[0])+','+str(line[1])+','+str(line[2])+','+str(colors[i])+','+str(seg[i])+',-100\n')
    f.close()
    
    #a scene done


# In[29]:


#.ply visualization
pcd_load=o3d_load(writepath+'.ply')
o3d_visualize(pcd_load.points,pcd_load.colors)


# In[30]:


#.txt verification
r=open(txt_path,'r')
a=np.genfromtxt(r, delimiter=",", usemask=True)
print(a.shape)
for i in a:
    print(i)


# In[ ]:





# In[ ]:




