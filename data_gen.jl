include("Potts.jl")
include("/home/abhijith/Dropbox/Code/GRISE_code/Potts_learning.jl")
using IterTools:product


p =15
q = 2
L = 6

Terms = Dict{}()
#The interaction strengths used in the paper are given below.
#J = [0.2598910820966015  -0.653715969944616  0.07787582451255748 -0.3704026001241134 0.0397234544123 0.23441234]
#For now we will generate them at random
J =  0.15*randn(L)
#The larger the size of these numbers , more smaples will be required to leran the model. This can be seen from the IT bound
println(J)

for r in 1:L
    M = pth_order_tensor(r)
    for i in 1:p-L+1
        index =  Tuple(i:i+r-1)
        Terms[index] =  J[r]*M
    end
end


F = FactorGraph([1:L...],p,q,Terms);


#Sample from model and save at path
n_samples = 10^6

s = raw_sampler_potts(F, n_samples, false)
samples=[2*a .- 3  for a in s]
sample_path ="samples.csv"
writedlm( sample_path,hcat(samples...))
#println("exact", countmap(samples))

#Generating another set of samples for testing the trained model.


s = raw_sampler_potts(F, 10^6, false)
samples2=[2*a .- 3  for a in s]
sample_path ="samples_test.csv"
writedlm( sample_path,hcat(samples2...))
