## purpose: compute the optimal cut-off values by maximizing the hazard ratios
# author: Hongming Xu
# email: mxu@ualberta.ca

library("readxl")
library(survival)
library(survminer)
library("forestplot")
library("xlsx")


rela_path='../../../'


# patient survivals
p_data<-read.csv(paste(rela_path,'data/kang_colon_data/','2020_01_08_Colon_MSI_262_survival.csv',sep=""))

pid<-vector()
futime<-vector()
fustat<-vector()
for (nn in 1:184) # excluding MSI patients
{
  pid_temp<-paste(as.character(p_data$patient_group[nn]),'_',as.character(p_data$Patient_ID[nn]),sep="")
  if (!is.na(p_data$ID[nn]))
  {
    pid<-c(pid,pid_temp)
    futime<-c(futime,as.numeric(p_data$OS[nn]))
    fustat<-c(fustat,as.character(p_data$OS_01[nn]))
  }
}

# tils density path
my_data<-read_excel(paste(rela_path,'data/pan_cancer_tils/feat_tils/yonsei_colon/threshold0.4/','til_density0.5.xlsx',sep=""))

univ_analysis1=TRUE


## 1) use each feature to divide patients into two groups, then plot km curves for univariate analysis
if (univ_analysis1==TRUE)
{
  #for (nn in c("feat0","feat1","feat2","feat3","feat4","feat5","feat6")) # 5 features
  for (nn in c("feat1","feat2","feat3","feat4"))
  {
    feat_v<-vector()
    for (pp in 1:length(pid))
    {
      ind<-which(pid[pp]==as.character(my_data$`patient id`))
      feat_v<-c(feat_v,as.numeric(my_data[nn][ind,1]))
    }
    
    ind_na<-which(is.na(feat_v))
    feat_v<-feat_v[-c(ind_na)]
    futime2<-futime[-c(ind_na)]
    fustat2<-fustat[-c(ind_na)]
    pid2<-pid[-c(ind_na)]
    
    #tt<-quantile(feat_v,0.34) # use median value to divide into high vs low
    #tt<-0.15
    
    features<-vector()
    cut_off_value<-vector()
    hazard_ratios<-vector()
    p_values<-vector()
    for (pc in seq(0.15,0.85,by=0.05))
    {
      tt<-quantile(feat_v,pc)
      plabel<-(feat_v>tt[1])
      plabel[plabel==TRUE]<-'High'
      plabel[plabel==FALSE]<-'Low'
      
      ## plot survival curves
      data_df<-data.frame("patientID"=Reduce(rbind,pid2))
      data_df$futime<-futime2
      data_df$fustat<-fustat2
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
                        xlab='Time in Months')+ggtitle("Yonsei Colon Cohort")
      
      ggsave(file=paste('2_',nn,'_',pc,".png",sep=""),print(survp),path=paste(rela_path,'data/pan_cancer_tils/km_curves/yonsei_os/',sep=""))
      
      rm(data_df)
      
    }
    
    df<-data.frame("features"=features,"cut off values"=cut_off_value,"hazard ratios"=hazard_ratios,'p values'=p_values)
    write.xlsx(df,paste(rela_path,'data/pan_cancer_tils/km_curves/yonsei_os/',nn,'.xlsx',sep=""),sheetName = "Sheet1", 
               col.names = TRUE, row.names = TRUE, append = FALSE)
    
  }
}
