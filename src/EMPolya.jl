function EMPolya(alpha0::AbstractArray,delta::AbstractArray,alpha::AbstractArray,Y::AbstractArray,TS::AbstractArray,p::AbstractArray,L::Int64,I::AbstractArray,J::Int64,K::Int64,EMNum::Int64,MMNum::Int64,BaseID::Int64=0,stopc::Float64=1e-4)
    mu=Array{Array{Float64}}(undef,L)
    MTS=zeros(Int64,L)
    MY=zeros(Int64,J,L)
    dLike=0
    dLikeNew=0
    MLike=zeros(Float64,K)
    lp=log.(p)

    #Average the initial delta
    if L>1
        delta[:,:,1:end.!=BaseID].=mean(delta[:,:,1:end.!=BaseID],dims=3)
        alpha=alpha0.+delta
    end

        @inbounds    for l in 1:L
            mu[l]=zeros(Float64,K,I[l])
            MTS[l]=maximum(TS[l])
            @inbounds    for i in 1:I[l]
                @inbounds for k in 1:K
                    MLike[k]=lp[k,l]+logpdf(DirichletMultinomial(TS[l][i],alpha[:,k,l]),view(Y[l],:,i))
                end
                m=maximum(MLike)
                @inbounds for k in 1:K
                    mu[l][k,i]=MLike[k]-m-log(sum(exp.(MLike[:].-m)))
                end
                dLike+=m+log(sum(exp.(MLike[:].-m)))

        end
        lp[:,l]=maximum(mu[l],dims=2).+log.(sum(exp.(mu[l].-maximum(mu[l],dims=2)),dims=2)).-log(I[l])
        @inbounds for j in 1:J
            MY[j,l]=maximum(Y[l][j,:])
        end
    end
    Newmu=copy(mu)
    Newlp=copy(lp)

    Flag=1
    @inbounds for t1 in 1:EMNum

        #For the first ten EM iterations, Recalculate and average the delta
        if 2<=t1<=10
            if L>1
                delta[:,:,1:end.!=BaseID].=mean(delta[:,:,1:end.!=BaseID],dims=3)
                alpha=alpha0.+delta
            end

            @inbounds    for l in 1:L
                @inbounds    for i in 1:I[l]
                    @inbounds for k in 1:K
                        MLike[k]=lp[k,l]+logpdf(DirichletMultinomial(TS[l][i],alpha[:,k,l]),view(Y[l],:,i))
                    end
                    m=maximum(MLike)
                    @inbounds for k in 1:K
                        Newmu[l][k,i]=MLike[k]-m-log(sum(exp.(MLike[:].-m)))
                    end
                    dLike+=m+log(sum(exp.(MLike[:].-m)))

            end

            end

        end

        #MM step
        alpha0, delta, alpha=MMPolya(alpha0,delta,alpha,Y,TS,MY,MTS,Newmu,L,I,J,K,MMNum)

        #Compute the updated likelihood
        dLikeNew=0
        @inbounds for l in 1:L
            @inbounds for i in 1:I[l]
                s=0
                @inbounds for k in 1:K
                    MLike[k]=lp[k,l]+logpdf(DirichletMultinomial(TS[l][i],alpha[:,k,l]),view(Y[l],:,i))
                end
                m=maximum(MLike)
                @inbounds for k in 1:K
                    Newmu[l][k,i]=MLike[k]-m-log(sum(exp.(MLike[:].-m)))
                end
                dLikeNew+=m+log(sum(exp.(MLike[:].-m)))
            end
            Newlp[:,l]=maximum(mu[l],dims=2).+log.(sum(exp.(mu[l].-maximum(mu[l],dims=2)),dims=2)).-log(I[l])
        end

        #Stop criteria
        if dLikeNew>dLike
            if abs((dLikeNew-dLike)/dLike)> stopc
                dLike=dLikeNew
                mu=copy(Newmu)
                lp=copy(Newlp)
            else
                Flag=0
                println(dLike)
                println(dLikeNew)
            end
        else
            Flag=0
            println("decreasing likelihood!")
            println(dLike)
            println(dLikeNew)
        end

        if(Flag==0)
            println("Niter=",t1)
            break
        end

    end

    #Determine cell label based on mu
    groups=mapslices(argmax, mu[1], dims=1)[:]
    if L>1
        @inbounds for l in 2:L
            groups=vcat(groups,mapslices(argmax, mu[l], dims=1)[:])
        end
    end

    return groups, dLikeNew, alpha0, delta, alpha, lp
end
