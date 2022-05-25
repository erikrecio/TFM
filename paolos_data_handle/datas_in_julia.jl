using DataFrames
using CSV

sp = [0. ,1., 0., 0.]

df = DataFrame()
#=
mus=[]
one_pointfunc=[]
two_pointfunc=[]
one_point_sum=[]
two_point_sum=[]
for i in data
    push!(mus,i[1])
    push!(one_pointfunc,i[2])
    push!(two_pointfunc,i[3])
    push!(one_point_sum,i[4])
    push!(two_point_sum,i[5])
end=#

#df = DataFrame(mu = mus, one_pointfunc = one_pointfunc,two_pointfunc=two_pointfunc,one_point_sum=one_point_sum,two_point_sum=two_point_sum );


df = DataFrame(sp = sp );
CSV.write("/Users/paolostornati/ICFO/Erik_thesis/TFM/paolos_data_handle/sp.csv", df)