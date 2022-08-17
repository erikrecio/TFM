using ITensors
using Random
using Optim
using BlackBoxOptim

using BenchmarkTools
using Dates
using Printf

#using OptimKit
#using Zygote

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
    write(f, "\np1_i = [")

    for j in 1:nsites

      orthogonalize!(ψ0,j)
      Sz_j = op("Sz", s, j)
      ψ0_dag_j = dag(prime(ψ0[j], "Site"))
      p = real.(round( 0.5 - scalar(ψ0_dag_j * Sz_j * ψ0[j]), digits = 3) )

      p_str = @sprintf "%4.3f" p

      if j != nsites
        write(f, "$p_str, ")
      else
        write(f, "$p_str")
      end

    end
    write(f, "]")
  end

  ############################################################
  # Print the entanglement entropy of the wave function ######
  ############################################################

  q_middle = trunc(Int, (nsites + 1)/2)

  orthogonalize!(ψ0, q_middle)
  U,S,V = svd(ψ0[q_middle], (linkind(ψ0, q_middle-1), siteind(ψ0, q_middle)))
  SvN = 0.0
  for n=1:dim(S, 1)
    p = S[n,n]^2
    SvN -= p * log(p)
  end

  return ψ0, SvN

end 

function optim_nelder(ψ0, nqubits0, nlayers, iter, qubit0_start, qubit0_end)

  ####################################
  # Optimization                ######
  ####################################

  nsites = length(ψ0)
  s = siteinds(ψ0)

  θ⃗₀ = 2π * rand(nsites * nlayers)
  rest = Optim.optimize(θ⃗ -> loss(θ⃗, ψ0, nqubits0, nlayers, qubit0_start, qubit0_end), θ⃗₀, NelderMead(), Optim.Options(iterations = iter, g_tol = 8e-7))
  
  ####################################
  # Print final wave function   ######
  ####################################

  open(name_file_sumup, "a") do f
    write(f, "\np1_f = [")

    for j in 1:nsites
      orthogonalize!(ψθ⃗::MPS,j)
      Sz_j = op("Sz", s, j)
      ψθ⃗_dag_j = dag(prime(ψθ⃗[j]::ITensor, "Site"))
      p = real.(round( 0.5 - scalar(ψθ⃗_dag_j * Sz_j * ψθ⃗[j]::ITensor), digits = 3) )

      p_str = @sprintf "%4.3f" p

      if j != nsites
        write(f, "$p_str, ")
      else
        write(f, "$p_str")
      end

    end
    write(f, "]")

    s_spaces_begin = "     , "^(qubit0_start - 1)

    if qubit0_end == nsites
      s_spaces_end = ""
      s_last = ""
      s_zeros = "0.000, "^(nqubits0 - 1) * "0.000"
    else
      s_spaces_end = "     , "^(nsites - qubit0_end - 1)
      s_last = "     "
      s_zeros = "0.000, "^(nqubits0)
    end
    

    write(f, "\nidea = [$s_spaces_begin$s_zeros$s_spaces_end$s_last]")
    write(f, "\ng_converged = $(rest.g_converged)")
    write(f, "\nmin_loss = $(rest.minimum)")
    write(f, "\niterations $(rest.iterations)/$iter")
  end

  open(name_file_plot, "a") do f
    write(f, "\n$(rest.g_converged) $(rest.minimum) $(rest.iterations)")
  end

  return nothing
end

function optim_black_box(ψ0, nqubits0, nlayers, iter, qubit0_start, qubit0_end)
  
  ####################################
  # Optimization                ######
  ####################################

  nsites = length(ψ0)
  s = siteinds(ψ0)

  θ⃗₀ = 2π * rand(nsites * nlayers)
  global rest = bboptimize(θ⃗ -> loss(θ⃗, ψ0, nqubits0, nlayers, qubit0_start, qubit0_end); SearchRange = (-3.15, 3.15), NumDimensions = nsites * nlayers, Method = :simultaneous_perturbation_stochastic_approximation)

  @show rest

  ####################################
  # Print final wave function   ######
  ####################################

  open(name_file_sumup, "a") do f
    write(f, "\np1_f = [")

    for j in 1:nsites
      orthogonalize!(ψθ⃗::MPS,j)
      Sz_j = op("Sz", s, j)
      ψθ⃗_dag_j = dag(prime(ψθ⃗[j]::ITensor, "Site"))
      p = real.(round( 0.5 - scalar(ψθ⃗_dag_j * Sz_j * ψθ⃗[j]::ITensor), digits = 3) )

      p_str = @sprintf "%4.3f" p

      if j != nsites
        write(f, "$p_str, ")
      else
        write(f, "$p_str")
      end

    end
    write(f, "]")

    s_spaces_begin = "     , "^(qubit0_start - 1)

    if qubit0_end == nsites
      s_spaces_end = ""
      s_last = ""
    else
      s_spaces_end = "     , "^(nsites - qubit0_end - 1)
      s_last = "     "
    end
    
    s_zeros = "0.000, "^(nqubits0)
    

    write(f, "\nidea = [$s_spaces_begin$s_zeros$s_spaces_end$s_last]")
    write(f, "\nstop_reason = $(rest.stop_reason) ")
    write(f, "\niterations $(rest.iterations)/$iter")
  end

  open(name_file_plot, "a") do f
    write(f, "\n$(rest.stop_reason) $(rest.iterations)")
  end


  #=
    method         :: String
    stop_reason    :: String
    iterations     :: Int64
    start_time     :: Float64
    elapsed_time   :: Float64
    parameters     :: AbstractDict{Symbol, Any}
    f_calls        :: Int64
    fit_scheme     :: FitnessScheme
    archive_output :: BlackBoxOptim.ArchiveOutput
    method_output  :: BlackBoxOptim.MethodOutput
  =#

  return nothing

end


function main()

  ####################################
  #   Parameters   ###################
  ####################################

  #Random.seed!(1234)

  #conf_        = [begin, end, runs,  step]
  conf_nsites   = [4,       4,    1,     0]
  conf_nqubits0 = [1,       1,    1,     0]
  conf_h        = [0.06, 0.18,    1,     0]
  conf_nlayers  = [1,       4,    1,     0]

  method = 1
  iter = 1000000000


  ####################################
  #   Code   #########################
  ####################################

  # Caculate Steps if missing ###

  if conf_nsites[4] == 0
    if conf_nsites[3] != 1
      conf_nsites[4] = (conf_nsites[2] - conf_nsites[1])/(conf_nsites[3] - 1) 
    else
      conf_nsites[4] = conf_nsites[2]
    end
  end
  
  if conf_h[4] == 0
    if conf_h[3] != 1
      conf_h[4] = (conf_h[2] - conf_h[1])/(conf_h[3] - 1)
    else
      conf_h[4] = conf_h[2]
    end
  end

  if conf_nqubits0[4] == 0
    if conf_nqubits0[3] != 1
      conf_nqubits0[4] = (conf_nqubits0[2] - conf_nqubits0[1])/(conf_nqubits0[3] - 1)
    else
      conf_nqubits0[4] = conf_nqubits0[2]
    end
  end
  
  if conf_nlayers[4] == 0
    if conf_nlayers[3] != 1
      conf_nlayers[4] = (conf_nlayers[2] - conf_nlayers[1])/(conf_nlayers[3] - 1)
    else
      conf_nlayers[4] = conf_nlayers[2]
    end
  end
  

  # Caculate Runs if missing ###

  if conf_nsites[3] == 0
    conf_nsites[3] = trunc(Int, (conf_nsites[2] - conf_nsites[1] + 1)/conf_nsites[4] + 1)
  end
  
  if conf_h[3] == 0
    conf_h[3] = (conf_h[2] - conf_h[1])/conf_h[4] + 1
  end

  if conf_nqubits0[3] == 0
    conf_nqubits0[3] = trunc(Int, (conf_nqubits0[2] - conf_nqubits0[1])/conf_nqubits0[4] + 1)
  end
  
  if conf_nlayers[3] == 0
    conf_nlayers[3] = trunc(Int, (conf_nlayers[2] - conf_nlayers[1])/conf_nlayers[4] + 1)
  end

  # Choose the directory to save the files
  dir_pc = "C:/Users/Nosgraph/Documents/GitHub/TFM/Results raw/"
  dir_lap = "/home/user/Documents/TFM/TFM/Results raw/"
  
  dir = dir_lap

  if ispath(dir_pc)
    dir = dir_pc
  else
    dir = dir_lap
  end

  # Name the sumup file 
  time_now = Dates.format(now(), "yy.mm.dd e, HH.MM.SS")
  global name_file_sumup = dir * time_now *  " - 0. Sumup.txt"
  

  # Name the plot file with the changes
  name_changes = "Time"
  
  if conf_nsites[3] != 1
    name_changes *= " vs nsites"
  else
    name_changes *= " vs nsites = $(conf_nsites[1])"
  end
  if conf_nqubits0[3] != 1
    name_changes *= " vs nqubits0"
  else
    name_changes *= " vs nqubits0 = $(conf_nqubits0[1])"
  end
  if conf_h[3] != 1
    name_changes *= " vs h"
  else
    name_changes *= " vs h = $(conf_h[1])"
  end
  if conf_nlayers[3] != 1
    name_changes *= " vs nlayers"
  else
    name_changes *= " vs nlayers = $(conf_nlayers[1])"
  end

  global name_file_plot = dir * time_now * " - " * name_changes * ".txt"

  # Write the headers in the files
  open(name_file_sumup, "a") do f
    write(f, "nsites = $conf_nsites\nnqubits0 = $conf_nqubits0\nh = $conf_h\nnlayers = $conf_nlayers\n\nchanges = $name_changes\niter = $iter\nmethod = $method\n\n")
  end

  open(name_file_plot, "a") do f
    write(f, "g_converged min_loss iter nsites nqubits0 h nlayers SvN time")
  end

  #initialize some variables
  ψ0 = 0
  SvN = 0
  nsites_2 = 0
  nqubits0_2 = 0
  h_2 = 999
  qubit0_start = 0
  qubit0_end = 0

  for nsites in range(conf_nsites[1], conf_nsites[2], step = conf_nsites[4])
    max_nqubits0 = min(nsites, conf_nqubits0[2])
    for h in range(conf_h[1], conf_h[2], step = conf_h[4])
      for nqubits0 in range(conf_nqubits0[1], max_nqubits0, step = conf_nqubits0[4])
        for nlayers in range(conf_nlayers[1], conf_nlayers[2], step = conf_nlayers[4])
    
          nsites = trunc(Int, nsites)
          nqubits0 = trunc(Int, nqubits0)
          nlayers = trunc(Int, nlayers)
        
          
          # Save the qubits in the middle of the state (the ones we will turn to 0)
          if nsites_2 != nsites || nqubits0_2 != nqubits0
            qubit0_start = trunc(Int, (nsites-nqubits0)/2 ) + 1
            qubit0_end = qubit0_start + nqubits0 - 1
          end

          # Calculate the ground state everytime there is a change
          if h_2 != h || nsites_2 != nsites
            ψ0, SvN = ground_state(nsites, h)
          end

          nsites_2 = nsites
          nqubits0_2 = nqubits0
          h_2 = h

          # Execute minimizer algorithm, save the time
          if method == 1
            time = @elapsed optim_nelder(ψ0, nqubits0, nlayers, iter, qubit0_start, qubit0_end)
          elseif method == 2
            time = @elapsed optim_black_box(ψ0, nqubits0, nlayers, iter, qubit0_start, qubit0_end)
          end

          #time = round(time, digits = 2)
        
          # Write the last data on the files
          open(name_file_sumup, "a") do f
            write(f, "\nh = $h")
            write(f, "\nnlayers = $nlayers")
            write(f, "\nSvN = $SvN")
            write(f, "\ntime = $time s\n")
          end

          open(name_file_plot, "a") do f
            write(f, " $nsites $nqubits0 $h $nlayers $SvN $time")
          end
        end
      end
    end
  end

  return nothing

end


for j in range(1, 1, step=1)
  main()
end




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

#= New Algorithms

  Simultaneous Perturbation Stochastic Approximation (SPSA)
    https://github.com/robertfeldt/BlackBoxOptim.jl

  Constrained Optimization by Linear Approximation (COBYLA)

=#
