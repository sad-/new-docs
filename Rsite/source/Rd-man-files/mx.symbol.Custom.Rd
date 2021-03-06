% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mxnet_generated.R
\name{mx.symbol.Custom}
\alias{mx.symbol.Custom}
\title{Custom:Apply a custom operator implemented in a frontend language (like Python).}
\usage{
mx.symbol.Custom(...)
}
\arguments{
\item{data}{NDArray-or-Symbol[]
Input data for the custom operator.}

\item{op.type}{string
Name of the custom operator. This is the name that is passed to `mx.operator.register` to register the operator.}

\item{name}{string, optional
Name of the resulting symbol.}
}
\value{
out The result mx.symbol
}
\description{
Custom operators should override required methods like `forward` and `backward`.
The custom operator must be registered before it can be used.
Please check the tutorial here: http://mxnet.io/faq/new_op.html.
}
\details{
Defined in src/operator/custom/custom.cc:L547
}

