% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ADMM_groupLasso_Log.R
\name{admmgrouplasso_ls}
\alias{admmgrouplasso_ls}
\title{Square Loss with group LASSO Penalty}
\usage{
admmgrouplasso_ls(X, y, group, lam, rho = 0.001)
}
\arguments{
\item{X}{An n by p design matrix}

\item{y}{The response vector with n entries}

\item{group}{The indicator vetor for grouping with n entries}

\item{lam}{The non-negative tuning parameter value}

\item{rho}{The penalty parameter in the augmented Lagrangian}
}
\value{
This function returns a list with four elements:
\item{beta}{The vector of regression coefficient estimates, with p-1 entries}
\item{total.iterations}{Number of iterations made}
}
\description{
Compute the group-LASSO penalized least square solutions
}
\examples{
\dontrun{
admmgrouplasso_ls(X=X[,-group], y=y, group=X[,group], lam=lam, rho=1e-3)}
}

