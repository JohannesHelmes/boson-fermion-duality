### generate the geometry for a equilateral triangle to obtain the pi/4 angle, a is the length of the kathete
function triangle(L, a)
    mask = zeros(Bool, (L,L))

    llc_x, llc_y = Int(round(L/2 - a/2 + 1)), Int(round(L/2 - a/2 + 1)) # lower left corner of the triangle
    for (n,i) in enumerate(collect(llc_x:llc_x + a -1))
        mask[i, llc_y:llc_y+n-1] = true
    end
    mask
end

### generate the geometry for a square of length a
function square(L, a)
    mask = zeros(Bool, (L,L))

    llc_x, llc_y = Int(round(L/2 - a/2 + 1)), Int(round(L/2 - a/2 + 1)) # lower left corner of the triangle
    mask[llc_x:llc_x+a-1,llc_y:llc_y+a-1] = true
    mask
end

### generate the geometry for a band winding around the torus with 8 corners with tan theta = -2, a is half the width of the band
function band_tanm2(L, a::Int)
    mask = zeros(Bool, (L,L))

    mask[:,1:2*a] = true
    m_x, m_y = Int(round(L/4 - a/2 + 1)), Int(round(a + 1)) # lower left corner of the triangle

    for i::Int in 0:round(a/2)-1
        mask[m_x-i:m_x+a+i-1,m_y+2*i] = false
        mask[m_x-i:m_x+a+i-1,m_y+2*i+1] = false
    end

    m_x, m_y = Int(round(3*L/4 - a/2 + 1)), a # lower left corner of the triangle
    for i::Int in 0:round(a/2)-1
        mask[m_x-i:m_x+a+i-1,m_y-2*i] = false
        mask[m_x-i:m_x+a+i-1,m_y-2*i-1] = false
    end
    mask
end

### generate the geometry for a band winding around the torus with 8 corners with tan theta = -1, a is the width of the band
function band_tanm1(L, a::Int)
    mask = zeros(Bool, (L,L))

    mask[:,1:a] = true
    m_x, m_y = Int(round(L/4 - a/2 + 1)), Int(round(a/2 + 1)) # lower left corner of the triangle

    for i::Int in 0:round(a/2)-1
        mask[m_x-i:m_x+a+i-1,m_y+i] = false
    end

    m_x, m_y = Int(round(3*L/4 - a/2 + 1)), Int(round(a/2)) # lower left corner of the triangle
    for i::Int in 0:round(a/2)-1
        mask[m_x-i:m_x+a+i-1,m_y-i] = false
    end
    mask
end

### generate the geometry for a band winding around the torus with 8 corners with tan theta = -1/2, a is the width of the band
function band_tanmh(L, a::Int)
    mask = zeros(Bool, (L,L))

    mask[:,1:a] = true
    m_x, m_y = Int(round(L/4 - a/2 + 1)), Int(round(a/2 + 1)) # lower left corner of the triangle

    for i::Int in 0:round(a/2)-1
        mask[m_x-2*i:m_x+a+2*i-1,m_y+i] = false
    end

    m_x, m_y = Int(round(3*L/4 - a/2 + 1)), Int(round(a/2)) # lower left corner of the triangle
    for i::Int in 0:round(a/2)-1
        mask[m_x-2*i:m_x+a+2*i-1,m_y-i] = false
    end
    mask
end

### generate the geometry for a parallelogram to obtain the 3pi/4 angle by subtraction of the pi/4 angle, a is the length of the non pixel side
function parallelo45(L, a::Int)
    mask = zeros(Bool, (L,L))

    llc_x, llc_y = Int(round(L/2 - a/2 + 1)), Int(round(L/2 - a)) # lower left corner of the triangle
    for (n,i) in enumerate(collect(llc_x:llc_x + a -1))
        mask[i, llc_y+n:llc_y+n+a-1] = true
    end
    mask
end

### generate the geometry for a parallelogram to obtain the tan theta = 1/2 angle by subtraction of the -1/2 angle, 
### a is the length of the non pixel side
function parallelotanh(L, a::Int)
    mask = zeros(Bool, (L,L))

    llc_x, llc_y = Int(round(L/2 - a/2 + 1)), Int(round(L/2 - 2*a)) # lower left corner of the triangle
    for (n,i) in enumerate(collect(llc_x:llc_x + a -1))
        mask[i, llc_y+2*n-1:llc_y+2*n-1+a-1] = true
    end
    mask
end

### generate the geometry for a parallelogram to obtain the tan theta = 2 angle by subtraction of the -2 angle, 
### a is the length of the non pixel side
function parallelotan2(L, a::Int)
    mask = zeros(Bool, (L,L))

    llc_x, llc_y = Int(round(L/2 - a + 1)), Int(round(L/2 - a)) # lower left corner of the triangle
    for (n,i) in enumerate(collect(llc_x:2:llc_x + 2*a -1))
        mask[i, llc_y+n:llc_y+n+a-1] = true
        mask[i+1, llc_y+n:llc_y+n+a-1] = true
    end
    mask
end

# Periodic boundary conditions in x direction, anti-periodic in y
function generate_ks(L)
    kx= 2*pi/L*collect(0:L-1)
    ky= pi/L*collect(1:2:2*L)
    kxgrid = repmat(kx,1, L)
    kygrid = repmat(ky', L)
    kxgrid , kygrid
end


# returns eigenvectors of the 2 band momentum space Hamiltonian
function k_eigenvectors!(kx::Array{Float64}, ky::Array{Float64}, sigx, sigy, sigz, evecs1, evecs2, L::Int)
    for i in 1:L
        for j in 1:L
            d = [sin(kx[i,j]), sin(ky[i,j]), 2 - cos(kx[i,j]) - cos(ky[i,j])]

            darr = d[1]*sigx + d[2]*sigy + d[3]*sigz
            F = eigfact!(darr)

            evecs1[i,j], evecs2[i,j] = F[:vectors][:,1]
        end
    end
end


function fillCmat_A!(Cmatrix_A::Array{Complex{Float64},2}, cdagc::Array{Complex{Float64},2}, cdagd::Array{Complex{Float64},2}, ddagc::Array{Complex{Float64},2}, ddagd::Array{Complex{Float64},2}, Arow::Array{Int64,1}, Acol::Array{Int64,1}, N::Int, L::Int)
    for (m,i) in enumerate(Arow)
        for (n,j) in enumerate(Acol)
            Cmatrix_A[m,n] = cdagc[1+mod(Arow[m]-Arow[n],L),1+mod(Acol[m]-Acol[n],L)]
            Cmatrix_A[m,n+N] = cdagd[1+mod(Arow[m]-Arow[n],L),1+mod(Acol[m]-Acol[n],L)]
            Cmatrix_A[m+N,n] = ddagc[1+mod(Arow[m]-Arow[n],L),1+mod(Acol[m]-Acol[n],L)]
            Cmatrix_A[m+N,n+N] = ddagd[1+mod(Arow[m]-Arow[n],L),1+mod(Acol[m]-Acol[n],L)]
        end
    end
end


# First create full correlation matrix, then reduce it to region A
function CmatA(L::Int, maskA::Array{Bool,2})
    kxg, kyg = generate_ks(L)

    evecs1 = similar(kxg, Complex{Float64})
    evecs2 = similar(kxg, Complex{Float64})
    sigx = [0 1; 1 0]
    sigy = [0 -1im; 1im 0]
    sigz = [1 0; 0 -1]
  
    k_eigenvectors!(kxg, kyg, sigx, sigy, sigz, evecs1, evecs2, L)
    #println(evecs2[1,1])

    #println(evecs1 .* conj(evecs1))
    #ev1ev = evecs1 .* conj(evecs1)
    cdagc = ifft(evecs1 .* conj(evecs1) )
    ddagd = ifft(evecs2 .* conj(evecs2) )

    ddagc = ifft(evecs1 .* conj(evecs2) )
    cdagd = ifft(evecs2 .* conj(evecs1) )

    #println(ddagd[1,2])

    #Now, reduce it to A
    Arow, Acol, trues = findnz(maskA)

    N=length(Arow)

    Cmatrix_A = Array{Complex{Float64}}(2*N, 2*N)

    fillCmat_A!(Cmatrix_A, cdagc, cdagd, ddagc, ddagd, Arow, Acol, N, L)
    #println(Cmatrix_A)

    Cmatrix_A
end

function Entanglement(alphas::Array{Int64}, L::Int, regionA::Array{Bool,2})
    corrA::Hermitian = Hermitian( CmatA(L, regionA) )
    #println(ishermitian(corrA))

    pA = abs(real(eigvals!(corrA)))

    sent=zeros(length(alphas))
    for (i,alpha) in enumerate(alphas)
        if alpha==1
            sent[i] = -1* sum(pA.*log(pA+1e-15))-1* sum((1-pA).*log(abs(1-pA+1e-15)))
        else
            sent[i] = (1/(1-alpha)) * sum(log((pA+1e-15).^alpha + abs(1-pA+1e-15).^alpha))
        end
    end
    sent
end


mult = 4
L=74:2:80

alphas = [1, 2, 3, 4]

for l in L
    #regionA = triangle(l*mult, l)
    #regionA = square(l*mult, l)
    #regionA = parallelo45(l*mult, l)
    #regionA = parallelotanh(l*mult, l)
    #regionA = parallelotan2(l*mult, l)
    #regionA = band_tanm2(l*mult, l)
    regionA = band_tanm1(l*mult, l)
    #regionA = band_tanmh(l*mult, l)
    #println( regionA)
    entropies = Entanglement(alphas, l*mult, regionA)
    println(l," ",entropies[1]," ",entropies[2]," ",entropies[3]," ",entropies[4]," ")
end


