
set JULIA_NUM_THREADS=4

@show Threads.nthreads()

i = Threads.Atomic{Int}(0)
ids = zeros(4)
old_is = zeros(4)

Threads.@threads for id in 1:4
    old_is[id] = Threads.atomic_add!(i, id)
    ids[id] = id
end

@show old_is
@show i
@show i[]
@show ids