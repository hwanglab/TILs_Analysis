## purpose: compute the optimal cut-off values by maximizing the hazard ratios
# analysis on tcga_coad stage ii and iii patients
# author: Hongming Xu, CCF, 2020
# email: mxu@ualberta.ca

library("readxl")
library(survival)
library(survminer)
library("forestplot")
library("xlsx")

rela_path='../../../'


# patient survivals
p_data<-read_excel(paste(rela_path,'data/tcga_coad_slide/','tcga_coad_read_kang.xlsx',sep=""))
p_data<-p_data[!duplicated(p_data$`Patient ID`),]

pid<-vector()
futime<-vector()
fustat<-vector()
for (nn in 1:length(p_data$`Patient ID`)) 
{
  pid_temp<-p_data$`Patient ID`[nn]
  pid<-c(pid,pid_temp)
  futime<-c(futime,as.numeric(p_data$`Disease Free (Months)`[nn]))
  fustat<-c(fustat,as.character(p_data$DFS_status_01[nn]))
  
}

# tils density path
my_data<-read_excel(paste(rela_path,'data/pan_cancer_tils/feat_tils/tcga_coad/threshold0.4/','til_density0.6.xlsx',sep=""))

# switches for different options
univ_analysis1=TRUE  # two-class km plots
univ_analysis2=TRUE   # three-class km plots
univ_analysis3=TRUE  # forest plot & univariate cxo proportional hazards analysis

## 1) use each feature to divide patients into two groups, then plot km curves for univariate analysis
if (univ_analysis1==TRUE)
{
  for (nn in c("feat0","feat1","feat2","feat3","feat4","feat5","feat6")) # 5 features
  {
    feat_v<-vector()
    for (pp in 1:length(pid))
    {
      ind<-which(pid[pp]==as.character(substr(my_data$`patient id`,1,12)))
      feat_v<-c(feat_v,as.numeric(my_data[nn][ind,1]))
    }
    
    ind_na<-which(is.na(feat_v))
    feat_v<-feat_v[-c(ind_na)]
    futime2<-futime[-c(ind_na)]
    fustat2<-fustat[-c(ind_na)]
    pid2<-pid[-c(ind_na)]
    
    ind_na2<-which(is.na(futime2))
    futime3<-futime2[-c(ind_na2)]
    fustat3<-fustat2[-c(ind_na2)]
    pid3<-pid2[-c(ind_na2)]
    feat_v2<-feat_v[-c(ind_na2)]
    
    #tt<-quantile(feat_v2,0.50) # use median value to divide into high vs low
    #tt<-0.15
    #print(tt)
    features<-vector()
    cut_off_value<-vector()
    hazard_ratios<-vector()
    p_values<-vector()
    for (pc in seq(0.15,0.85,by=0.05))
    {
      tt<-quantile(feat_v,pc)
      plabel<-(feat_v2>tt[1])
      plabel[plabel==TRUE]<-'High'
      plabel[plabel==FALSE]<-'Low'
      
      ## plot survival curves
      data_df<-data.frame("patientID"=Reduce(rbind,pid3))
      data_df$futime<-futime3
      data_df$fustat<-fustat3
      data_df$plabel<-plabel
      
      data_df$futime<-as.numeric(as.character(data_df$futime)) # note that: must be numeric type
      data_df$fustat<-as.numeric(as.character(data_df$fustat)) # note that: must be numeric type
      
      
      # fit survival data using the kaplan-Meier method
      surv_object<-Surv(time=data_df$futime,event=data_df$fustat)
      # surv_object
      
      fit1<-survfit(surv_object~plabel,data=data_df)
      # summary(fit1)
      # 
      
      # univariate cox analysis for a certain category
      res.cox<-coxph(surv_object~plabel,data=data_df)
      res<-summary(res.cox)
      hr<-res$coefficients[2]
      #p_value<-res$coefficients[5] # wald test pvalue
      p_value<-res$sctest[3]        # the same pvalue as plotted on KM curves
      
      cut_off_value<-c(cut_off_value,tt[1])
      hazard_ratios<-c(hazard_ratios,hr)
      p_values<-c(p_values,p_value)
      features<-c(features,nn)
      
      #setEPS()
      #postscript("whatever.eps")
      
      survp<-ggsurvplot(fit1,pval = TRUE,
                        risk.table = TRUE,
                        legend=c(0.8,0.2),
                        #legend.labs=c("High (42)","Low (19)"),
                        legend.title="Categories",
                        xlab='Time in Months')+ggtitle("TCGA_COAD Cohort")
      ggsave(file=paste('2_',nn,'_',pc,".png",sep=""),print(survp),path=paste(rela_path,'data/pan_cancer_tils/km_curves/tcga_dfs/',sep=""))
      
      
      rm(data_df)
    }
    
    df<-data.frame("features"=features,"cut off values"=cut_off_value,"hazard ratios"=hazard_ratios,'p values'=p_values)
    write.xlsx(df,paste(rela_path,'data/pan_cancer_tils/km_curves/tcga_dfs/',nn,'.xlsx',sep=""),sheetName = "Sheet1", 
               col.names = TRUE, row.names = TRUE, append = FALSE)
    
  }
  
 
}
