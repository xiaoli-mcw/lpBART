mc.lpbart <- function(
    x.train, y.train, x.test = matrix(0.0, 0L, 0L), 
    sparse=FALSE, theta=0, omega=1,
    a=0.5, b=1, augment=FALSE, rho=NULL,
    xinfo=matrix(0.0,0,0), usequants=FALSE,
    cont=FALSE, rm.const=TRUE,
    k = 2.0, ## BEWARE: do NOT use k for other purposes below
    power = 2.0, base = 0.95, mub=NULL, g=NULL,
    ntree=50L, numcut=100L,
    ndpost=1000L, nskip=500L,
    keepevery=5L, printevery=100L,
    keeptrainfits=TRUE, transposed=FALSE,
##    treesaslists=FALSE,
    mc.cores = 2L, nice = 19L,
    seed = 99L
)
{
    if(.Platform$OS.type!='unix')
        stop('parallel::mcparallel/mccollect do not exist on windows')

    RNGkind("L'Ecuyer-CMRG")
    set.seed(seed)
    parallel::mc.reset.stream()

    temp.xtrain <- cbind(1, x.train)
    xl.train <- temp.xtrain[, qr(temp.xtrain)$pivot[seq_len(qr(temp.xtrain)$rank)]]
    full <- data.frame(y=y.train,xl.train[,-1])
    #glfml <- as.formula(paste("y",paste(names(full)[-1],collapse="+"),sep="~"))
    fit <- glm(y~., data=full, family=binomial(link="probit"))
    if(is.null(mub)) mub <- fit$coef
    rm(full,fit,temp.xtrain,xl.train)
        

    if(!transposed) {
        temp = bartModelMatrix(x.train, numcut, usequants=usequants,
                               cont=cont, xinfo=xinfo, rm.const=rm.const)
        x.train = t(temp$X)
        numcut = temp$numcut
        xinfo = temp$xinfo
        ## if(length(x.test)>0)
        ##     x.test = t(bartModelMatrix(x.test[ , temp$rm.const]))
        if(length(x.test)>0) {
            x.test = bartModelMatrix(x.test)
            x.test = t(x.test[ , temp$rm.const])
        }
        rm.const <- temp$rm.const
        rm(temp)
    }

    mc.cores.detected <- detectCores()

    if(mc.cores>mc.cores.detected) mc.cores <- mc.cores.detected
        ## warning(paste0('The number of cores requested, mc.cores=', mc.cores,
        ##                ',\n exceeds the number of cores detected via detectCores() ',
        ##                'which yields ', mc.cores.detected, ' .'))

    mc.ndpost <- ceiling(ndpost/mc.cores)
    ## mc.ndpost <- ((ndpost %/% mc.cores) %/% keepevery)*keepevery

    ## while(mc.ndpost*mc.cores<ndpost) mc.ndpost <- mc.ndpost+keepevery

    ## mc.nkeep <- mc.ndpost %/% keepevery

    for(i in 1:mc.cores) {
        parallel::mcparallel({psnice(value=nice);
                  lpbart(x.train=x.train, y.train=y.train, x.test=x.test,
                        sparse=sparse, theta=theta, omega=omega,
                        a=a, b=b, augment=augment, rho=rho,
                        xinfo=xinfo,
                        k=k, power=power, base=base, mub=mub, g=g,
                        ntree=ntree, numcut=numcut,
                        ndpost=mc.ndpost, nskip=nskip, keepevery=keepevery,
                        ## nkeeptrain=mc.nkeep, nkeeptest=mc.nkeep,
                        ## nkeeptestmean=mc.nkeep, nkeeptreedraws=mc.nkeep,
                        printevery=printevery, transposed=TRUE)},
                        ##treesaslists=treesaslists)},
                  silent=(i!=1))
                  ## to avoid duplication of output
                  ## capture stdout from first posterior only
    }

    post.list <- parallel::mccollect()

    post <- post.list[[1]]

    if(mc.cores==1 | attr(post, 'class')!='lpbart') return(post)
    else {
        if(class(rm.const)[1]!='logical') post$rm.const <- rm.const

        post$ndpost <- mc.cores*mc.ndpost
        
        p <- nrow(x.train[post$rm.const, ])
        ##p <- nrow(x.train[ , post$rm.const])

        ## if(length(rm.const)==0) rm.const <- 1:p
        ## post$rm.const <- rm.const

        old.text <- paste0(as.character(mc.ndpost), ' ', as.character(ntree),
                           ' ', as.character(p))
        old.stop <- nchar(old.text)

        post$treedraws$trees <- sub(old.text,
                                    paste0(as.character(post$ndpost), ' ',
                                           as.character(ntree), ' ',
                                           as.character(p)),
                                    post$treedraws$trees)

        keeptestfits <- length(x.test)>0

        for(i in 2:mc.cores) {
            if(keeptrainfits) {
                post$yhat.train <- rbind(post$yhat.train,
                                         post.list[[i]]$yhat.train)
                post$prob.train <- rbind(post$prob.train,
                                         post.list[[i]]$prob.train)
                post$beta.train <- rbind(post$beta.train,
                                         post.list[[i]]$beta.train)
                post$sigma.mu <- rbind(post$sigma.mu,
                                       post.list[[i]]$sigma.mu)
                post$kdraw <- rbind(post$kdraw,
                                    post.list[[i]]$kdraw)
            }

            if(keeptestfits) {
                post$yhat.test <- rbind(post$yhat.test,
                                        post.list[[i]]$yhat.test)
                post$prob.test <- rbind(post$prob.test,
                                        post.list[[i]]$prob.test)
            }

            post$varcount <- rbind(post$varcount, post.list[[i]]$varcount)
            post$varprob <- rbind(post$varprob, post.list[[i]]$varprob)

            post$treedraws$trees <- paste0(post$treedraws$trees,
                                           substr(post.list[[i]]$treedraws$trees, old.stop+2,
                                                  nchar(post.list[[i]]$treedraws$trees)))

            ## if(treesaslists) post$treedraws$lists <-
            ##                      c(post$treedraws$lists, post.list[[i]]$treedraws$lists)
        }

        ## if(length(post$yhat.train.mean)>0)
        ##     post$yhat.train.mean <- apply(post$yhat.train, 2, mean)

        ## if(length(post$yhat.test.mean)>0)
        ##     post$yhat.test.mean <- apply(post$yhat.test, 2, mean)

        if(length(post$prob.train.mean)>0)
            post$prob.train.mean <- apply(post$prob.train, 2, mean)

        if(length(post$prob.test.mean)>0)
            post$prob.test.mean <- apply(post$prob.test, 2, mean)

        post$varcount.mean <- apply(post$varcount, 2, mean)
        post$varprob.mean <- apply(post$varprob, 2, mean)

        attr(post, 'class') <- 'lpbart'

        return(post)
    }
}
