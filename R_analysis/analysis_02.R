## purpose: perform multi-variable cox regression analysis and generate forestplot figure
# author Hongming Xu
# email: mxu@ualberta.ca

##--------example-----------##
# ref: http://www.sthda.com/english/wiki/cox-proportional-hazards-model
#      https://cran.r-project.org/web/packages/forestplot/vignettes/forestplot.html

library("survival")
library("survminer")
library("forestplot")

head(lung)

## univariate cox regression
res.cox<-coxph(Surv(time,status)~sex, data=lung)

## to apply the univariate coxph function to multiple covariates at once:
covariates <- c("age", "sex",  "ph.karno", "ph.ecog", "wt.loss")
univ_formulas <- sapply(covariates,
                        function(x) as.formula(paste('Surv(time, status)~', x)))

univ_models <- lapply( univ_formulas, function(x){coxph(x, data = lung)})
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
           zero = 1,
           cex  = 2,
           lineheight = "auto",
           xlab = "Risk for Event")

##multivariate cox regression analysis
res.cox<-coxph(Surv(time,status)~age+sex+ph.ecog,data=lung)
summary(res.cox)

# visualizing the estimated distribution of survival times
# note that: there is error in the orginal example
ggsurvplot(survfit(res.cox,data=lung),palette="#2E9FDF",ggtheme=theme_minimal())

# create the new data
sex_df<-with(lung,data.frame(sex=c(1,2),
                             age=rep(mean(age,na.rm = TRUE),2),
                             ph.ecog=c(1,1)
))

fit<-survfit(res.cox,newdata = sex_df,data=lung)
ggsurvplot(fit,conf.int = TRUE,legend.labs=c('Sex=1','Sex=2'),ggtheme = theme_minimal())
