using ITensors
using Random
using Optim
using BenchmarkTools
using Dates
using Printf

hello

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


function loss(θ⃗, ψ0, nqubits0, nlayers, qubit0_start, qubit0_end)
  nsites = length(ψ0)
  s = siteinds(ψ0)
  
  𝒰θ⃗ = variational_circuit(nsites, nlayers, θ⃗)
  Uθ⃗ = ops(𝒰θ⃗, s)
  global ψθ⃗ = apply(Uθ⃗, ψ0; cutoff=1e-8)

  p1 = 0

  for j in qubit0_start:qubit0_end
    orthogonalize!(ψθ⃗::MPS,j)
    Sz_j = op("Sz", s, j)
    ψθ⃗_dag_j = dag(prime(ψθ⃗[j]::ITensor, "Site"))
    p1 += 0.5 - scalar(ψθ⃗_dag_j * Sz_j * ψθ⃗[j]::ITensor)
  end

  return real.(p1)/nqubits0

end


function ground_state(nsites, h)

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

  open(name_file_sumup, "a") do f
    write(f, "p1_i = [")

    for j in 1:nsites

      orthogonalize!(ψ0,j)
      Sz_j = op("Sz", s, j)
      ψ0_dag_j = dag(prime(ψ0[j], "Site"))
      p = real.(round( 0.5 - scalar(ψ0_dag_j * Sz_j * ψ0[j]), digits = 3) )

      if j != nsites
        write(f, "$p, ")
      else
        write(f, "$p")
      end

    end
    write(f, "]")
  end

  return ψ0

end 

function optim_nelder(ψ0, nqubits0, nlayers, iter, qubit0_start, qubit0_end)

  ####################################
  # Optimization                ######
  ####################################

  nsites = length(ψ0)
  s = siteinds(ψ0)

  θ⃗₀ = 2π * rand(nsites * nlayers)
  rest = optimize(θ⃗ -> loss(θ⃗, ψ0, nqubits0, nlayers, qubit0_start, qubit0_end), θ⃗₀, NelderMead(), Optim.Options(iterations = iter, g_tol = 1e-6))
  

  ####################################
  # Print final wave function   ######
  ####################################

  open(name_file_sumup, "a") do f
    write(f, "\n\ng_converged = $(rest.g_converged) ")
    write(f, "\niterations $(rest.iterations)/$iter")
    write(f, "\np1_f = [")

    for j in 1:nsites
      orthogonalize!(ψθ⃗::MPS,j)
      Sz_j = op("Sz", s, j)
      ψθ⃗_dag_j = dag(prime(ψθ⃗[j]::ITensor, "Site"))
      p = real.(round( 0.5 - scalar(ψθ⃗_dag_j * Sz_j * ψθ⃗[j]::ITensor), digits = 3) )

      if j != nsites
        write(f, "$p, ")
      else
        write(f, "$p")
      end

    end
    write(f, "]")
  end

  return nothing
end


function main()

  #Random.seed!(1234)

  nsites = 5
  nqubits0 = 2
  
  iter = 3000
  i_blocs = 1

  h = 0.5
  nlayers = 3



  # We measure the qubits in the middle of the state
  qubit0_start = trunc(Int, (nsites-nqubits0)/2 ) + 1
  qubit0_end = qubit0_start + nqubits0 - 1

  dir_pc = "D:/Users/Usuario/Documents/1 Master Quantum Physics and Technology/TFM/Repo GitHub/TFM/Results raw/"
  dir_lap = "/home/user/Documents/TFM/TFM/Results raw/"
  
  dir = dir_lap

  if ispath(dir_pc)
    dir = dir_pc
  else
    dir = dir_lap
  end

  time_now = Dates.format(now(), "dd.mm.yy e, HH.MM.SS")
  global name_file_sumup = dir * time_now *  " - 0. Sumup.txt"

  name_file_prova1 = dir * time_now * " - Prova 1 - Time vs iter.txt"
  name_file_prova2 = dir * time_now *  " - Prova 2 - Time vs nsites.txt"
  name_file_prova3 = dir * time_now *  " - Prova 3 - Time vs nqubits0.txt"
  name_file_prova4 = dir * time_now *  " - Prova 4 - Time vs nlayers.txt"
  name_file_prova5 = dir * time_now *  " - Prova 5 - Time vs h.txt"
  
  open(name_file_sumup, "a") do f
    write(f, "h = $h\nnsites = $nsites\n\nnqubits0 = $nqubits0\nqubit0_start = $qubit0_start\nqubit0_end = $qubit0_end\n\nnlayers = $nlayers\niter = $iter\n\n")
  end


  i_begin = trunc(Int, iter/i_blocs)
  i_end = i_begin * i_blocs
  i_step = i_begin

  ψ0 = 0
  nsites_2 = 0
  h_2 = 999

  for i in range(i_begin, i_end, step = i_step)
    
    if h_2 != h || nsites_2 != nsites
      ψ0 = ground_state(nsites, h)
      h_2 = h
      nsites_2 = nsites
    end

    time = @elapsed optim_nelder(ψ0, nqubits0, nlayers, i, qubit0_start, qubit0_end)

    time = round(time, digits = 2)

    open(name_file_sumup, "a") do f
      write(f, "\ntime = $time s")
    end

    open(name_file_prova1, "a") do f
      write(f, "$i $time\n")
    end

  end

  return nothing

end

main()



#= Questions:

  1. When is the bond dimension defined? (lambda > cutoff=1e-8?)

=#

#= Saved code:

  write(f, @sprintf("nsites = %i nqubits0 = %i nlayers = %i iter = %i\n",nsites, nqubits0, nlayers, iter))
  p1_f = Float64[]
  
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