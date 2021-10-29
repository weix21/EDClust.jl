function fitPolya(Count::AbstractArray,Sample::AbstractArray,InitVal::AbstractArray,EMNum::Int64=100,MMNum::Int64=5,BaseID::Int64=0,stopc::Float64=1e-4)
    Sample=convert(Array{Int64}, Sample)
    L=length(unique(Sample))
    I=counts(Sample)
    J=size(InitVal)[1]
    K=size(InitVal)[2]

    Y=Array{Array{Int64}}(undef,L)
    for l in 1:L
        Y[l]=Count[:,(Sample.==l)[:]]
    end

    TS=Array{Array{Int64}}(undef,L)
    for l in 1:L
        TS[l]=sum(Y[l],dims=1)[:]
    end

    Salpha0=InitVal[:,:]
    Salpha0[Salpha0.<1e-10].=1e-10
    Sdelta=zeros(Float64,J,K,L)
    Salpha=zeros(Float64,J,K,L)

    for l in 1:L
        if l!=BaseID
            Sdelta[:,:,l].=1e-5
        end
        Salpha[:,:,l]=Salpha0+Sdelta[:,:,l]
    end

    Sp=1/K.*ones(K,L)
    return EMPolya(Salpha0,Sdelta,Salpha,Y,TS,Sp,L,I,J,K,EMNum,MMNum,BaseID,stopc)
end
