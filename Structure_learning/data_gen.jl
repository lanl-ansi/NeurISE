include("Potts.jl")
using IterTools:product


function pth_order_tensor(r)
    a =  Tuple([2 for i in 1:r])
    M =  zeros(a)
    for i in CartesianIndices(M)
        s = prod(2*[Tuple(i)...] .- 3)
        M[i] = s
    end
    return M
end


n =15
q = 2


Terms = Dict{}()



M =  pth_order_tensor(2)

for i in 1:n-1
    Terms[(i,i+1)] = (0.3 + rand())*M

end

Terms[(1,n)] = (0.3 + rand())*M

Terms[(1,3,5,7,9)] = 0.5*pth_order_tensor(5)




F = FactorGraph([2,5],n,q,Terms);



n_samples =10^6

s = raw_sampler_potts(F, n_samples, false)
samples=[2*a .- 3  for a in s]
sample_path ="saved_model/samples.csv"
writedlm( sample_path,hcat(samples...))
