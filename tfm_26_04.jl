using ITensors
using Random
using Optim
using BenchmarkTools
using Dates
using Printf


function ising_hamiltonian(nsites; h)
  ℋ = OpSum()
  for j in 1:(nsites - 1)
    ℋ += -1, "Z", j, "Z", j + 1
  end
  for j in 1:nsites
    ℋ += h, "X", j
  end
  return ℋ
end

# A layer of the circuit we want to optimize
function layer(nsites, θ⃗)
  RY_layer = [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:nsites]
  CX_layer = [("CX", (n, n + 1)) for n in 1:(nsites - 1)]
  return [RY_layer; CX_layer]
end

# The variational circuit we want to optimize
function variational_circuit(nsites, nlayers, θ⃗)
  range = 1:nsites
  circuit = layer(nsites, θ⃗[range])
  for n in 1:(nlayers - 1)
    circuit = [circuit; layer(nsites, θ⃗[range .+ n * nsites])]
  end
  return circuit
end


function loss(θ⃗, ψ0, nqubits0, nlayers)
  nsites = length(ψ0)
  s = siteinds(ψ0)
  
  𝒰θ⃗ = variational_circuit(nsites, nlayers, θ⃗)
  Uθ⃗ = ops(𝒰θ⃗, s)
  global ψθ⃗ = apply(Uθ⃗, ψ0; cutoff=1e-8)

  p1 = 0

  # We measure the qubits in the middle of the state
  qubit0_start = trunc(Int, (nsites-nqubits0)/2 ) + 1
  qubit0_end = qubit0_start + nqubits0 - 1

  for j in qubit0_start:qubit0_end
    orthogonalize!(ψθ⃗::MPS,j)
    Sz_j = op("Sz", s, j)
    ψθ⃗_dag_j = dag(prime(ψθ⃗[j]::ITensor, "Site"))
    p1 += 0.5 - scalar(ψθ⃗_dag_j * Sz_j * ψθ⃗[j]::ITensor)
  end

  return real.(p1)

end


function ground_state(nsites, nqubits0, nlayers, h, iter)

  ####################################
  # Calculate Ground State      ######
  ####################################

  s = siteinds("Qubit", nsites)
  ψ0 = MPS(s, "0")
  #ψ0 = randomMPS(ComplexF64, s; linkdims=bondim)
  
  ℋ = ising_hamiltonian(nsites; h=h)
  H = MPO(ℋ, s)
  
  sweeps = Sweeps(5)
  setmaxdim!(sweeps, 10)
  e_dmrg, ψ0 = dmrg(H, ψ0, sweeps)

  ####################################
  # Print initial wave function ######
  ####################################


  p1_i = Float64[]


  f = open(name_file_sumup, "w")
  write(f, "nsites = $nsites nqubits0 = $nqubits0 nlayers = $nlayers iter = $iter\n")
  write(f, "p1_i = [")
  #write(f, @sprintf("nsites = %i nqubits0 = %i nlayers = %i iter = %i\n",nsites, nqubits0, nlayers, iter))

  for j in 1:nsites

    orthogonalize!(ψ0,j)
    Sz_j = op("Sz", s, j)
    ψ0_dag_j = dag(prime(ψ0[j], "Site"))
    p = real.(round( 0.5 - scalar(ψ0_dag_j * Sz_j * ψ0[j]), digits = 3) )
    push!(p1_i, p)

    if j != nsites
      write(f, "$p, ")
    else
      write(f, "$p")
    end

  end
  write(f, "]")
  close(f)

  @show p1_i

  return ψ0

end 

function optim_nelder(ψ0, nqubits0, nlayers, iter)

  ####################################
  # Optimization                ######
  ####################################

  nsites = length(ψ0)
  s = siteinds(ψ0)

  θ⃗₀ = 2π * rand(nsites * nlayers)
  rest = optimize(θ⃗ -> loss(θ⃗, ψ0, nqubits0, nlayers), θ⃗₀, NelderMead(), Optim.Options(iterations = iter))
  

  ####################################
  # Print final wave function   ######
  ####################################

  @show nsites
  @show nqubits0
  @show nlayers
  @show iter

  pf1 = Float64[]

  for j in 1:nsites
    orthogonalize!(ψθ⃗::MPS,j)
    Sz_j = op("Sz", s, j)
    ψθ⃗_dag_j = dag(prime(ψθ⃗[j]::ITensor, "Site"))
    push!(pf1, real.(round( 0.5 - scalar(ψθ⃗_dag_j * Sz_j * ψθ⃗[j]::ITensor), digits = 3) ))
  end

  @show pf1
  @show rest
  
  #=
  @show rest

  @show rest.ls_success
  @show min = rest.minimum
  @show θ⃗op = rest.minimizer
  
  @show rest.iterations
  @show rest.f_calls

  @show rest.g_converged
  @show rest.g_abstol
  @show rest.iteration_converged
  @show rest.stopped_by
  
  @show rest.time_limit
  @show rest.time_run
  =#

  return nothing
end


function main()

  #Random.seed!(1234)

  nsites = 5
  nqubits0 = 2
  iter = 10

  h = 0.5
  nlayers = 3

  nsites_2 = 0
  h_2 = 999
  
  time_now = Dates.format(now(), "e, dd.mm.yy HH.MM.SS")
  dir = "D:/Users/Usuario/Documents/1 Master Quantum Physics and Technology/TFM/JuliaFiles/"
  global name_file_sumup = dir * time_now * ".txt"

  if h_2 != h || nsites_2 != nsites
    ψ0 = ground_state(nsites, nqubits0, nlayers, h, iter)
    h_2 = h
    nsites_2 = nsites
  end

  time = @elapsed optim_nelder(ψ0, nqubits0, nlayers, iter)

  @show time

  return nothing

end

main()



#= Questions:
1. When is the bond dimension defined? (lambda > cutoff=1e-8?)

=#