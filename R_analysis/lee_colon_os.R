## purpose: perform univariate variable analysis, e.g., tils densities -> patient survivials (os)
# analysis on lee_colon patient slides
# author: Hongming Xu, CCF, 2020
# email: mxu@ualberta.ca

# excel dates computation web link: 
# https://support.office.com/en-us/article/calculate-the-difference-between-two-dates-8235e7c9-b430-44ca-9425-46100a162f38
# https://www.ablebits.com/office-addins-blog/2015/03/26/excel-convert-text-date/
# https://www.ablebits.com/office-addins-blog/2015/06/10/excel-date-functions/

library("readxl")
library(survival)
library(survminer)
library("forestplot")

rela_path='../../../'


# patient survivals
p_data<-read_excel(paste(rela_path,'data/lee_colon_data/','Colorectal cancer dataset.xlsx',sep=""))
pid0<-paste(p_data$`S no (primary)`,p_data$`Sub no (T)...10`,sep='')
pid<-gsub('#','-',pid0)

#p_data<-p_data[!duplicated(p_data$`Patient ID`),]

#pid<-vector()
futime<-vector()
fustat<-vector()
for (nn in 1:length(pid)) 
{
  temp_stat<-as.character(p_data$`Expire`[nn])
  fustat<-c(fustat,temp_stat)
  
  futime<-c(futime,as.numeric(p_data$`Overall Survival (days)`[nn]))
  
}

# tils density path
my_data<-read_excel(paste(rela_path,'data/pan_cancer_tils/feat_tils/lee_colon/threshold0.4/','til_density0.5.xlsx',sep=""))

# switches for different options
univ_analysis1=TRUE  # two-class km plots
univ_analysis2=FALSE   # three-class km plots
univ_analysis3=TRUE  # forest plot & univariate cxo proportional hazards analysis

## 1) use each feature to divide patients into two groups, then plot km curves for univariate analysis
if (univ_analysis1==TRUE)
{
  for (nn in c("feat0","feat1","feat2","feat3","feat4","feat5","feat6","feat7")) # 5 features
  {
    feat_v<-vector()
    for (pp in 1:length(pid))
    {
      ## remove the leading 0 to match patient id together
      temp<-unlist(strsplit(pid[pp],'-'))
      temp[2]<-toString(as.numeric(temp[2]))
      temp_pid<-paste(temp,collapse = '-')
      
      ind<-which(temp_pid==as.character(my_data$`patient id`))
      feat_v<-c(feat_v,as.numeric(my_data[nn][ind,1]))
    }
    
    ind_na<-which(is.na(feat_v))
    feat_v<-feat_v[-c(ind_na)]
    futime2<-futime[-c(ind_na)]
    fustat2<-fustat[-c(ind_na)]
    pid2<-pid[-c(ind_na)]
    
    ind_na2<-which(is.na(futime2))
    if (length(ind_na2)>0)
    {
      futime3<-futime2[-c(ind_na2)]
      fustat3<-fustat2[-c(ind_na2)]
      pid3<-pid2[-c(ind_na2)]
      feat_v2<-feat_v[-c(ind_na2)]
    }
    else
    {
      futime3<-futime2
      fustat3<-fustat2
      pid3<-pid2
      feat_v2<-feat_v
    }
    
    fustat3[fustat3=='N']<-'0'
    fustat3[fustat3=='Y']<-'1'
    
    #tt<-quantile(feat_v2,0.34) # use median value to divide into high vs low
    tt<-0.15
    print(tt)
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
    
    #setEPS()
    #postscript("whatever.eps")
    
    survp<-ggsurvplot(fit1,pval = TRUE,
                      risk.table = TRUE,
                      legend=c(0.8,0.2),
                      #legend.labs=c("High (42)","Low (19)"),
                      legend.title="Categories",
                      xlab='Time in Days')+ggtitle("st.mary Colon Cohort")
    ggsave(file=paste('2_',nn,".png",sep=""),print(survp),path='./lee_os/')
    
    rm(data_df)
  }
}

## 2) three-levels KM plots
if (univ_analysis2==TRUE)
{
  for (nn in c("feat0","feat1","feat2","feat3","feat4","feat5","feat6")) # 5 features
  {
    feat_v<-vector()
    for (pp in 1:length(pid))
    {
      ## remove the leading 0 to match patient id together
      temp<-unlist(strsplit(pid[pp],'-'))
      temp[2]<-toString(as.numeric(temp[2]))
      temp_pid<-paste(temp,collapse = '-')
      
      ind<-which(temp_pid==as.character(my_data$`patient id`))
      feat_v<-c(feat_v,as.numeric(my_data[nn][ind,1]))
    }
    
    ind_na<-which(is.na(feat_v))
    feat_v<-feat_v[-c(ind_na)]
    futime2<-futime[-c(ind_na)]
    fustat2<-fustat[-c(ind_na)]
    pid2<-pid[-c(ind_na)]
    
    ind_na2<-which(is.na(futime2))
    if (length(ind_na2)>0)
    {
      futime3<-futime2[-c(ind_na2)]
      fustat3<-fustat2[-c(ind_na2)]
      pid3<-pid2[-c(ind_na2)]
      feat_v2<-feat_v[-c(ind_na2)]
    }
    else
    {
      futime3<-futime2
      fustat3<-fustat2
      pid3<-pid2
      feat_v2<-feat_v
    }
    
    fustat3[fustat3=='N']<-'0'
    fustat3[fustat3=='Y']<-'1'
    
    ttL<-quantile(feat_v2,0.333) # use median value to divide into high vs low
    ttH<-quantile(feat_v2,0.666)
    
    plabel<-cut(feat_v2,breaks=c(-1,ttL,ttH,Inf),labels=c("Low","Mid","High"))
    
    
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
    
    #setEPS()
    #postscript("whatever.eps")
    
    survp<-ggsurvplot(fit1,pval = TRUE,
                      risk.table = TRUE,
                      legend=c(0.8,0.2),
                      #legend.labs=c("High (42)","Low (19)"),
                      legend.title="Categories",
                      xlab='Time in Days')+ggtitle("st.mary Colon Cohort")
    ggsave(file=paste('3_',nn,".png",sep=""),print(survp),path='./lee_os/')
    
    rm(data_df)
  }
}


## 3) use univariate coxph function to perform univarate analysis
for (pp in 1:length(pid))
{
  ## remove the leading 0 to match patient id together
  temp<-unlist(strsplit(pid[pp],'-'))
  temp[2]<-toString(as.numeric(temp[2]))
  temp_pid<-paste(temp,collapse = '-')
  pid[pp]<-temp_pid
}

if (univ_analysis3==TRUE)
{
  my_pid<-my_data$`patient id`

  futime4<-vector()
  fustat4<-vector()
  for (pp in 1:length(my_pid))
  {
    ind<-which(my_pid[pp]==pid)
    if (length(ind)>0)  # in case no patients in the p_data
    {
      futime4<-c(futime4,futime[ind[1]])
      fustat4<-c(fustat4,fustat[ind[1]])
    }
    else
    {
      futime4<-c(futime4,NA)
      fustat4<-c(fustat4,NA)
    }

  }

  fustat4[fustat4=='N']<-'0'
  fustat4[fustat4=='Y']<-'1'

  my_data$futime<-as.numeric(futime4)
  my_data$fustat<-as.numeric(fustat4)

  ind_na<-which(is.na(my_data$futime))
  if (length(ind_na)>0)
  {
    my_data2<-my_data[-ind_na,] # remove rows with NA values
  }
  else
  {
    my_data2<-my_data
  }



  ## to apply the univariate coxph function to multiple covariates at once:
  covariates <- c("feat0", "feat1",  "feat2", "feat3", "feat4","feat5","feat6","feat7")
  univ_formulas <- sapply(covariates,
                          function(x) as.formula(paste('Surv(futime, fustat)~', x)))

  univ_models <- lapply( univ_formulas, function(x){coxph(x, data = my_data2)})
  # Extract data
  univ_results <- lapply(univ_models,
                         function(x){
                           x <- summary(x)
                           p.value<-signif(x$wald["pvalue"], digits=2)
                           wald.test<-signif(x$wald["test"], digits=2)
                           beta<-signif(x$coef[1], digits=2);#coeficient beta
                           HR <-signif(x$coef[2], digits=2);#exp(beta)
                           HR.confint.lower <- signif(x$conf.int[,"lower .95"], 2)
                           HR.confint.upper <- signif(x$conf.int[,"upper .95"],2)
                           #HR <- paste0(HR, " (",
                           #              HR.confint.lower, "-", HR.confint.upper, ")")
                           res<-c(beta, HR, HR.confint.lower, HR.confint.upper, wald.test, p.value)
                           #names(res)<-c("beta", "HR (95% CI for HR)", "wald.test",
                           #               "p.value")
                           names(res)<-c("beta", "HR", "HR.confint.lower", "HR.confint.upper", "wald.test",
                                         "p.value")
                           return(res)
                           #return(exp(cbind(coef(x),confint(x))))
                         })
  res <- t(as.data.frame(univ_results, check.names = FALSE))
  df<-as.data.frame(res)
  row_names<-list(row.names(df),
                  sprintf("p=%.3f",df$p.value))

  forestplot(row_names,
             fn.ci_norm = fpDrawCircleCI, # change to circles
             df$HR, # coef
             df$HR.confint.lower,
             df$HR.confint.upper,
             graph.pos = 2, # graph on the second column, p on the last
             zero = 0,
             cex  = 2,
             lineheight = "auto",
             xlab = "Risk for Event")
}


t=0


