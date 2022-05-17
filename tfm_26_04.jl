using ITensors
using Random
using Optim
using BenchmarkTools


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
    orthogonalize!(ψθ⃗,j)
    Sz_j = op("Sz", s, j)
    ψθ⃗_dag_j = dag(prime(ψθ⃗[j], "Site"))
    p1 += 0.5 - scalar(ψθ⃗_dag_j * Sz_j * ψθ⃗[j])
  end

  return real.(p1)

end


function main(nsites, nqubits0, nlayers, h, iter)

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

  pi1 = []
  pf1 = []

  for j in 1:nsites
    orthogonalize!(ψ0,j)
    Sz_j = op("Sz", s, j)
    ψ0_dag_j = dag(prime(ψ0[j], "Site"))
    push!(pi1, real.(round( 0.5 - scalar(ψ0_dag_j * Sz_j * ψ0[j]), digits = 3) ))
  end

  @show nsites
  @show nqubits0
  @show nlayers
  @show iter

  @show pi1


  ####################################
  # Optimization                ######
  ####################################
  
  θ⃗₀ = 2π * rand(nsites * nlayers)
  #lv = loss(θ⃗₀)

  rest = optimize(θ⃗ -> loss(θ⃗, ψ0, nqubits0, nlayers), θ⃗₀, NelderMead(), Optim.Options(iterations = iter))
  θ⃗op = rest.minimizer
  min = rest.minimum
  

  ####################################
  # Print final wave function   ######
  ####################################

  
  for j in 1:nsites
    orthogonalize!(ψθ⃗,j)
    Sz_j = op("Sz", s, j)
    ψθ⃗_dag_j = dag(prime(ψθ⃗[j], "Site"))
    push!(pf1, real.(round( 0.5 - scalar(ψθ⃗_dag_j * Sz_j * ψθ⃗[j]), digits = 3) ))
  end

  @show pf1
  @show rest
  
  #=
  @show rest

  @show rest.ls_success
  @show rest.minimum
  @show rest.minimizer
  
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



#option: Random.seed!(1234)
nsites = 5
nqubits0 = 2
iter = 1000

h = 0.5
nlayers = 3


#@btime or @time
@time main(nsites, nqubits0, nlayers, h, iter)


#= Questions:
1. When is the bond dimension defined? (lambda > cutoff=1e-8?)


=#