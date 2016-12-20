#group/block soft thresholding operator
groupsoft=function(x,t){
  max(1-t/sqrt(sum(x^2)),0)*x  #sqrt(t(x)%*%x)
}

#order group index
ordergroup=function(x){
  x[order(x$group),]
}

#beta update using Newton Raphson
b_update=function(y,X,beta,alpha,omega,rho){
  # y=as.matrix(y)
  a=0.1
  b=0.5  # control step size in newton method.
  toler=1e-2
  maxiter=100
  f=function(beta){sum(log(1+exp(-y*(X%*%beta))))+(rho/2)*sum((beta-alpha+omega)^2)}
  f1=log(1+exp(-t(as.matrix(y))%*%(X%*%beta)))+(rho/2)*sum((beta-alpha+omega)^2)
  n=nrow(X)
  p=ncol(X)
  I=diag(p)
  C=-t(y)%*%X #calculate for only once to speed up.
  j=0
  t=0.01
  dfx=Inf
  diff=Inf
  # while (abs(dfx)>toler | j<maxiter)
  while (j<maxiter){
    j=j+1
    g=t(C)%*%(exp(C%*%beta)/(1+exp(C%*%beta))) +rho*(beta-alpha+omega) #gradient
    H=t(C)%*% diag(exp(C%*%beta)/((1+exp(C%*%beta))^2))%*% C + rho*I #Hessian
    dx=solve(H)%*%g
    dfx=-t(g) %*% dx #Newton decrement
    # while(f(beta-dx)>(f(beta)+a*t*dfx)) {t=b*t} #backtracking
    # beta.old=beta
    beta=beta-t*dx
    # diff=beta-beta.old
    #
    # print(j)
    # print(diff)
  }
  return(list("beta"=beta,"iter"=j,"diff"=sum(abs(diff))))
}

#' @title Logistic Loss with group LASSO Penalty
#' @description Compute the group-LASSO penalized minimum logistic loss solutions
#' @param X An n by p design matrix
#' @param y The response vector with n entries
#' @param group The indicator vetor for grouping with n entries
#' @param lam The non-negative tuning parameter value
#' @param rho The penalty parameter in the augmented Lagrangian
#' @return This function returns a list with two elements:
#' \item{beta}{The vector of regression coefficient estimates, with p-1 entries}
#' \item{total.iterations}{Number of iterations made}
#' @examples \dontrun{
#' admmgrouplasso_log(X=X[,-group], y=y, group=X[,group], lam=lam, rho=1e-3)}
#' @export admmgrouplasso_log

admmgrouplasso_log=function(X,y,group,lam,rho=1e-3){
  n=length(y)
  p=ncol(X)
  g=as.data.frame(table(group)) #grouping info
  size=g[,2]
  omega=rep(0,p)
  # alpha=rep(0,p)
  alpha_g=vector("list",nrow(g)) #group wise alpha
  for(j in 1:nrow(g)){
    alpha_g[[j]]=rep(0,g[j,2])
  }
  alpha=rep(0,p)

  maxit=1000 #max iter
  k=0
  beta=rep(0,p)
  iterating=TRUE
  fval=-Inf

  while(iterating){
    k=k+1

    beta=b_update(y,X,beta,alpha,omega,rho)$beta #update beta

    for(j in 1:nrow(g)){
      alpha_g[[j]]=groupsoft(beta[which(group==j)]+omega[which(group==j)],sqrt(size[j])*lam/rho)
    }
    temp.alpha=alpha_g[[1]]

    for(j in 2:nrow(g)){ #start from 2 since already has 1st value.
      temp.alpha=c(temp.alpha,alpha_g[[j]])
    }

    alpha=temp.alpha #update alpha

    omega=omega+beta-alpha #update omega

    gp=0 #group_penalized term
    for(j in 1:nrow(g)){
      gp=gp+sqrt(size[j])*sqrt(sum((beta[which(group==j)])^2))
    }

    old.fval=fval
    fval=sum(log(1+exp(-y*(X%*%beta))))+lam*gp

    if((k>maxit)|abs(fval-old.fval)<1e-3){iterating=FALSE}
  }
  beta=beta*p #times p because in the newton part, I forgot to sum over p for gradient.
  return(list("beta"=beta,"total.iterations"=k))
}


#soft thresholding function
soft=function(x,t){
  return(sign(x)*max(abs(x)-t,0))
}

#beta update objective function
b_update_fun=function(y,X,beta,alpha,omega,rho){
  return(sum(log(1+exp(-y*(X%*%beta))))+(rho/2)*sum((beta-alpha+omega)^2))
  # return(log(1+exp(-t(as.matrix(y))%*%(X%*%beta)))+(rho/2)*sum((beta-alpha+omega)^2))
}

#beta update using Newton Raphson
b_update=function(y,X,beta,alpha,omega,rho){
  # y=as.matrix(y)
  a=0.1
  b=0.5  # control step size in newton method.
  toler=1e-2
  maxiter=100
  f=function(beta){sum(log(1+exp(-y*(X%*%beta))))+(rho/2)*sum((beta-alpha+omega)^2)}
  f1=log(1+exp(-t(as.matrix(y))%*%(X%*%beta)))+(rho/2)*sum((beta-alpha+omega)^2)
  n=nrow(X)
  p=ncol(X)
  I=diag(p)
  C=-t(y)%*%X #calculate for only once to speed up.
  j=0
  t=0.01
  dfx=Inf
  diff=Inf
  # while (abs(dfx)>toler | j<maxiter)
  while (j<maxiter){
    j=j+1
    g=t(C)%*%(exp(C%*%beta)/(1+exp(C%*%beta))) +rho*(beta-alpha+omega) #gradient
    H=t(C)%*% diag(exp(C%*%beta)/((1+exp(C%*%beta))^2))%*% C + rho*I #Hessian
    dx=solve(H)%*%g
    dfx=-t(g) %*% dx #Newton decrement
    # while(f(beta-dx)>(f(beta)+a*t*dfx)) {t=b*t} #backtracking
    # beta.old=beta
    beta=beta-t*dx
    # diff=beta-beta.old
    #
    # print(j)
    # print(diff)
  }
  return(list("beta"=beta,"iter"=j,"diff"=sum(abs(diff))))
}

#' @title Logistic Loss with LASSO Penalty
#' @description Compute the LASSO penalized minimum logistic loss solutions
#' @param X An n by p design matrix
#' @param y The response vector with n entries
#' @param lam The non-negative tuning parameter value
#' @param rho The penalty parameter in the augmented Lagrangian
#' @return This function returns a list with four elements:
#' \item{beta}{The vector of regression coefficient estimates, with p-1 entries}
#' \item{total.iterations}{Number of iterations made}
#' @examples \dontrun{
#' admmlasso_log(X=X, y=y, lam=lam, rho=1e-3)}
#' @export admmlasso_log

admmlasso_log=function(y,X,lam,rho=1e-3){
  n=length(y)
  p=ncol(X)
  omega=rep(0,p)
  alpha=rep(0,p)
  beta=rep(0,p)

  k=0
  maxit=1000
  iterating=TRUE
  fval=-Inf

  while(iterating){
    k=k+1
    # beta=optim(rep(0.1,p),b_update_fun,y=y,X=X,alpha=alpha,omega=omega,rho=rho,method="L-BFGS-B",
    #            lower=rep(-100,p),upper=rep(100,p))$par
    beta=b_update(y,X,beta,alpha,omega,rho)$beta
    alpha=soft(x=(beta+omega),t=lam/rho)
    omega=omega+beta-alpha

    old.fval=fval
    fval=sum(log(1+exp(-y*(X%*%beta))))+lam*sum(abs(beta))
    if((k>maxit)|abs(fval-old.fval)<1e-3){iterating=FALSE}
  }
  return(list("beta"=p*beta,"total.iterations"=k))
}


groupsoft=function(x,t){
  max(1-t/sqrt(sum(x^2)),0)*x  #sqrt(t(x)%*%x)
}

ordergroup=function(x){
  x[order(x$group),]
}

#' @title Square Loss with group LASSO Penalty
#' @description Compute the group-LASSO penalized least square solutions
#' @param X An n by p design matrix
#' @param y The response vector with n entries
#' @param group The indicator vetor for grouping with n entries
#' @param lam The non-negative tuning parameter value
#' @param rho The penalty parameter in the augmented Lagrangian
#' @return This function returns a list with four elements:
#' \item{beta}{The vector of regression coefficient estimates, with p-1 entries}
#' \item{total.iterations}{Number of iterations made}
#' @examples \dontrun{
#' admmgrouplasso_ls(X=X[,-group], y=y, group=X[,group], lam=lam, rho=1e-3)}
#' @export admmgrouplasso_ls

admmgrouplasso_ls=function(X,y,group,lam,rho=1e-3){
  n=length(y)
  p=ncol(X)
  g=as.data.frame(table(group)) #grouping info
  size=g[,2]
  omega=rep(0,p)
  # alpha=rep(0,p)
  alpha_g=vector("list",nrow(g)) #group wise alpha
  for(j in 1:nrow(g)){
    alpha_g[[j]]=rep(0,g[j,2])
  }
  alpha=rep(0,p)

  I=diag(p)
  maxit=100000 #max iter
  k=0
  beta=rep(NA,p)
  iterating=TRUE
  fval=-Inf
  m=qr.solve(t(X)%*%X+rho*I)
  while(iterating){
    k=k+1
    beta=m%*%(t(X)%*%y+rho*(alpha-omega))

    for(j in 1:nrow(g)){
      alpha_g[[j]]=groupsoft(beta[which(group==j)]+omega[which(group==j)],sqrt(size[j])*lam/rho)
    }

    temp.alpha=alpha_g[[1]]
    for(j in 2:nrow(g)){ #start from 2 since already has 1st value.
      temp.alpha=c(temp.alpha,alpha_g[[j]])
    }

    alpha=temp.alpha
    omega=omega+beta-alpha

    gp=0 #group_penalized term
    for(j in 1:nrow(g)){
      gp=gp+sqrt(size[j])*sqrt(sum((beta[which(group==j)])^2))
    }

    old.fval=fval
    fval=0.5*sum((y-X%*%beta)^2)+lam*gp
    if((k>maxit)|abs(fval-old.fval)<1e-7){iterating=FALSE}
  }
  return(list("beta"=beta,"total.iterations"=k))
}


soft=function(x,t){
  return(sign(x)*max(abs(x)-t,0))
}

#' @title Square Loss with LASSO Penalty
#' @description Compute the LASSO penalized least square solutions
#' @param X An n by p design matrix
#' @param y The response vector with n entries
#' @param lam The non-negative tuning parameter value
#' @param rho The penalty parameter in the augmented Lagrangian
#' @return This function returns a list with four elements:
#' \item{beta}{The vector of regression coefficient estimates, with p-1 entries}
#' \item{total.iterations}{Number of iterations made}
#' @examples \dontrun{
#' admmlasso_ls(X=X, y=y, lam=lam, rho=1e-3)}
#' @export admmlasso_ls

admmlasso_ls=function(y,X,lam,rho=1e-3){
  Xc = scale(X[,-1], scale = FALSE)
  yc = scale(y, scale = FALSE)
  ybar = mean(y)
  Xbar = colMeans(Xc)


  n=length(y)
  p=dim(Xc)[2]
  omega=rep(0,p)
  alpha=rep(0,p)
  I=diag(p)
  maxit=100000 #max iter
  # maxrho=5
  k=0
  beta=rep(NA,p)
  iterating=TRUE
  fval=-Inf
  m=qr.solve(t(Xc)%*%Xc+rho*I) #solve it only for once to boost the speed.
  while(iterating){
    k=k+1
    beta=m%*%(t(Xc)%*%yc+rho*(alpha-omega))
    alpha=soft(x=beta+omega,t=lam/rho)
    omega=omega+beta-alpha
    #rho=min(maxrho,rho*1.1)
    old.fval=fval
    fval=0.5*t(yc-Xc%*%beta)%*%(yc-Xc%*%beta)+lam*sum(abs(beta))
    if((k>maxit)|abs(fval-old.fval)<1e-7){iterating=FALSE}
    #if(sum(abs(fval-old.fval))<1e-4){iterating=FALSE}
  }
  beta0=ybar-Xbar%*%beta
  return(list("beta0"=beta0,"beta"=beta,"total.iterations"=k,"fval"=fval,"old.fval"=old.fval))
}
