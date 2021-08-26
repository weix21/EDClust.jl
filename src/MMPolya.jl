function MMPolya(alpha0::AbstractArray,delta::AbstractArray,alpha::AbstractArray,Y::AbstractArray,TS::AbstractArray,MY::AbstractArray,MTS::AbstractArray,mu::AbstractArray,L::Int64,I::AbstractArray,J::Int64,K::Int64,MMNum::Int64)
    Newdelta=similar(delta)
    Newalpha0=similar(alpha0)
    for t2 in 1:MMNum
        DS=zeros(J,K,L)
        DS2=zeros(J,K,L)
        DS1=zeros(K,L)
    @inbounds  for l in 1:L
                    @inbounds for k in 1:K
                        salpha=sum(view(alpha,:,k,l))

                        n2=maximum(view(mu[l],k,:)[TS[l].>=1])
                        @inbounds for c1 in 0:MTS[l]-1
                            DS1[k,l]+=sum(exp.(view(mu[l],k,:)[TS[l].>=c1+1].-n2))/(salpha+c1)
                        end
                        DS1[k,l]=n2+log(DS1[k,l])

                        @inbounds   for j in 1:J
                            if length(view(mu[l],k,:)[Y[l][j,:].>=1])==0
                                DS2[j,k,l]=-10000
                                DS[j,k,l]=0
                            else
                                n3=maximum(view(mu[l],k,:)[Y[l][j,:].>=1])
                                @inbounds   for c2 in 0:MY[j,l]-1
                                    DS2[j,k,l]+=sum(exp.(view(mu[l],k,:)[Y[l][j,:].>=c2+1].-n3))/(view(alpha,j,k,l).+c2)
                                end
                                DS2[j,k,l]=n3+log(DS2[j,k,l])
                                DS[j,k,l]=exp(DS2[j,k,l]-DS1[k,l])
                             end

                        end
                    end
                end
        Newdelta=delta.*DS
        Newalpha0=alpha0.*(sum(exp.(DS2),dims=3)./sum(exp.(DS1),dims=2)')
        delta=copy(Newdelta)
        alpha0=copy(Newalpha0)
        alpha0[alpha0.<1e-100].=1e-100
        for l in 1:L
            alpha[:,:,l]=alpha0+delta[:,:,l]
        end
    end
    alpha0, delta, alpha
end
