'''
purpose: quantify prediction map entropy

author: Hongming Xu
email: mxu@ualberta.ca
'''
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from skimage import transform

rela_path='../../../'

def shannon_entropy_bin(X,b=0.1,vmin=0.0,vmax=1.0):
    # shannon's entropy
    # input: X - numpy array or list

    a = np.arange(vmin,vmax+b,b)

    for i, xv in enumerate(X):
        temp = abs(xv - a)
        ind = np.argmin(temp)
        X[i] = a[ind]

    X_uni = np.unique(X)
    X_uni_size = len(X_uni)

    P = np.zeros(X_uni_size)

    for i in range(X_uni_size):
        P[i] = sum(X == X_uni[i])

    P = P / len(X)
    # Compute the Normalized Entropy
    H_norm = -sum(P * np.log2(P)) / (np.log2(X_uni_size)+1e-4)
    H = -sum(P * np.log2(P))

    return H_norm, H

if __name__=='__main__':

    # testing example
    #x=[0.4,0.37,0.7,0.8,0.90,0.91,0.93]
    #H_norm, H=shannon_entropy_bin(x,b=0.1)
    #print(H_norm)
    #print(H)

    tcga_stad=False
    classic_stad=True

    if tcga_stad==True:
        msi_path=rela_path + 'data/tcga_stad_slide/predictions_TCGA_STAD_analysis_binary_v1 (2)/'
        df_msi=pd.read_csv(msi_path+'patient_level_results.csv')
        til_path=rela_path+'data/tcga_stad_slide/tumor_msi_tils_maps/til_density_of_tumor.csv'
        df_til=pd.read_csv(til_path)
        imgs_id=df_til['img_id'].tolist()
        tild1=df_til['til_density'].tolist()
        tild2=df_til['til_density_v2'].tolist()

        surv_path=rela_path+'data/tcga_stad_slide/TCGA_STAD_master_table.xlsx'
        df_surv=pd.read_excel(surv_path)
        pid=df_surv['Patient ID'].tolist()
        pid_st=df_surv['Overall Survival (Months)'].tolist()
        pid_ss=df_surv['Overall Survival Status'].tolist()

        output_path=rela_path + 'data/tcga_stad_slide/'
        imgs = df_msi['Image_ID'].tolist()
    elif classic_stad==True:
        msi_path=rela_path+'data/CLASSIC_stomach_cancer_image/prediction_heatmaps_CLASSIC_Stomach_Cancer_Image_v1/predictions_CLASSIC_Stomach_Cancer_Image_analysis_v1/'
        df_msi = pd.read_csv(msi_path + 'patient_level_results_CLASSIC_Stomach_Cancer_Image.csv')

        df_cc = pd.read_csv(
            rela_path + 'data/CLASSIC_stomach_cancer_image/' + 'CLASSIC_cohort_validation_20201006_MSI.csv')
        imgs = df_cc['slide_id'].tolist()

        imgs_id2 = df_msi['Image_ID'].tolist()

        til_path=rela_path+'data/CLASSIC_stomach_cancer_image/til_maps/LEICA/'

        output_path = rela_path + 'data/CLASSIC_stomach_cancer_image/'

    else:
        raise RuntimeError('undefined option...')

    entropy_v11=[]
    entropy_v12=[]
    entropy_v21=[]
    entropy_v22=[]
    til_density_v1=[]
    til_density_v2=[]
    os_months=[]
    os_status=[]

    MSI_TD_masked = []
    MSI_TD_masked_global_entropy_10_bins = []
    MSI_TD_masked_global_entropy_20_bins = []

    for i, temp in enumerate(imgs):
        img=glob.glob(msi_path+str(temp)+'*.png')
        if len(img)==1:
            I=plt.imread(img[0])
            I1=I[:,:,0]
            msi_v=I1[I1>0]

            ent_v11, ent_v12=shannon_entropy_bin(msi_v.copy(),b=1.0)
            ent_v21, ent_v22=shannon_entropy_bin(msi_v.copy(),b=0.1)

            entropy_v11.append(ent_v11)
            entropy_v12.append(ent_v12)
            entropy_v21.append(ent_v21)
            entropy_v22.append(ent_v22)

            if tcga_stad==True:
                id=img[0].split('\\')[-1][0:23]

                if id in imgs_id:
                    ind = imgs_id.index(id)
                    til_density_v1.append(tild1[ind])
                    til_density_v2.append(tild2[ind])
                else:
                    til_density_v1.append('NA')
                    til_density_v2.append('NA')

                id2 = img[0].split('\\')[-1][0:12]
                if id2 in pid:
                    ind = pid.index(id2)
                    os_months.append(pid_st[ind])
                    os_status.append(pid_ss[ind])
                else:
                    os_months.append('NA')
                    os_status.append('NA')
            elif classic_stad==True:
                id=img[0].split('\\')[-1].split('_')[0]+'_'+img[0].split('\\')[-1].split('_')[1]
                if id[0]!='C':
                    id=id.split('_')[0]
                til_img=img=glob.glob(til_path+id+'_gray.png')
                I2=plt.imread(til_img[0])
                tumor_mask=(I1>0)
                til_mask=(I2[:,:,0]>0.5)
                tumor_mask=transform.resize(tumor_mask,til_mask.shape, order=0)
                til_tumor_ratio=np.sum(np.logical_and(til_mask,tumor_mask))/np.sum(tumor_mask)
                til_density_v1.append(til_tumor_ratio)

                ind = imgs_id2.index(id)
                MSI_TD_masked.append(df_msi['MSI_TD_masked'].tolist()[ind])
                MSI_TD_masked_global_entropy_10_bins.append(df_msi['MSI_TD_masked_global_entropy_10_bins'].tolist()[ind])
                MSI_TD_masked_global_entropy_20_bins.append(df_msi['MSI_TD_masked_global_entropy_20_bins'].tolist()[ind])
            else:
                raise RuntimeError('undefined option...')

        else:
            if classic_stad==True:
                entropy_v11.append('NA')
                entropy_v12.append('NA')
                entropy_v21.append('NA')
                entropy_v22.append('NA')
                til_density_v1.append('NA')
                MSI_TD_masked.append('NA')
                MSI_TD_masked_global_entropy_10_bins.append('NA')
                MSI_TD_masked_global_entropy_20_bins.append('NA')

    if tcga_stad==True:
        df_msi['entropy_v11'] = entropy_v11
        df_msi['entropy_v12'] = entropy_v12
        df_msi['entropy_v21'] = entropy_v21
        df_msi['entropy_v22'] = entropy_v22
        df_msi['til_density_v1'] = til_density_v1
        df_msi['til_density_v2'] = til_density_v2
        df_msi['os_months'] = os_months
        df_msi['os_status'] = os_status
        df_msi.to_csv(output_path + 'tcga_stad_survival_table.csv')
    elif classic_stad==True:
        df_cc['entropy_v11'] = entropy_v11
        df_cc['entropy_v12'] = entropy_v12
        df_cc['entropy_v21'] = entropy_v21
        df_cc['entropy_v22'] = entropy_v22
        df_cc['til_density_v1'] = til_density_v1
        df_cc['MSI_TD_masked'] = MSI_TD_masked
        df_cc['MSI_TD_masked_global_entropy_10_bins'] = MSI_TD_masked_global_entropy_10_bins
        df_cc['MSI_TD_masked_global_entropy_20_bins'] = MSI_TD_masked_global_entropy_20_bins
        df_cc.to_csv(output_path + 'classic_stad_master_table.csv')
    else:
        raise RuntimeError('undefined option...')
