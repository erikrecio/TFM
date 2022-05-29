using ITensors

function create_state(; L, d)

    # Creation of a Random State of dim = d^L (tensor T)
    dim = d^L

    sites = []
    for j in 1:L
        s = string(j)
        i_site = Index(d,"s"*s)
        push!(sites, i_site)
    end

    T = randomITensor(ComplexF64, sites)
    
    norm = scalar(T*dag(T))

    T /= sqrt(norm)

    @show scalar(T*dag(T))

    return T
    r=1
end

function main(; L, d)

    T = create_state(; L, d)

    #@show size(T)
    #@show inds(T)[1]

    Q = []
    R = []

    Qj, Rj = qr(T,inds(T)[1])
    push!(Q, Qj)
    push!(R, Rj)

    for j in 2:L
        Qj, Rj = qr(R[j-1],inds(R[j-1])[1:2])
        push!(Q, Qj)
        push!(R, Rj)
    end

    #Q1,R1 = qr(T,inds(T)[1])
    #Q2,R2 = qr(R1,inds(R1)[1:2])
    #Q3,R3 = qr(R2,inds(R2)[1:2])

    @show T
    
    T2 = Q[1]*Q[2]*Q[3]
    @show T2
    @show R[3]

    # Pq R[3] = 1 o R[3] = -1??? com pot ser que doni -1 si se suposa que és el mòdul de l'estat?
end

main(; L=3, d=2)