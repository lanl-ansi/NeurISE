using StatsBase
using Distributions
using DelimitedFiles
using Combinatorics: with_replacement_combinations

import LinearAlgebra
import LinearAlgebra: diag
import Statistics: mean







mutable struct FactorGraph{}
    order_list::Array{Int64,1}
    variable_count::Int
    n_alphabets::Int
    terms::Dict{Tuple,Array} # TODO, would be nice to have a stronger tuple type
    #TODO In the sanity check make sure that an kth order hyper edge is mapped to  a k rank tensor
    #variable_names::Union{Vector{String}, Nothing}
    #FactorGraph(a,b,c,d,e) = check_model_data(a,b,c,d,e) ? new(a,b,c,d,e) : error("generic init problem")
end











permutations(items, order::Int; asymmetric::Bool = false) = sort(permutations([], items, order, asymmetric))

function permutations(partial_perm::Array{Any,1}, items, order::Int, asymmetric::Bool)
    """
    All possible permutations of a given size.
    If asymmetric is false, then it returns combinations of items of the given order
    If asymmetric is true it returns all possible tuples of the size given by order from items
    """
    if order == 0
        return [tuple(partial_perm...)]
    else
        perms = []
        for item in items
            if !asymmetric && length(partial_perm) > 0
                if partial_perm[end] >= item
                    continue
                end
            end
            perm = permutations(vcat(partial_perm, item), items, order-1, asymmetric)
            append!(perms, perm)
        end
        return perms
    end
end







function raw_sampler_potts(H::FactorGraph, n_samples::Int, centered::Bool)
    """
    Given the FactorGraph, return samples according to its Gibbs distribution

    """

    n = H.variable_count
    q = H.n_alphabets
    n_config = q^n
    configs = [ digits(i,base=q, pad=n) .+ 1 for i = 0:n_config-1]
    weights = [ exp(Energy_Potts(K, H, centered)) for K in configs ]
    #print(configs, weights/sum(weights))
    raw_samples =  wsample(configs, weights, n_samples)
    return raw_samples

   end

function Energy_Potts(state::Array{Int64,  1},H::FactorGraph, cent::Bool)
    """
    Given a state and a FactorGraph, return its energy
    """
    q =  H.n_alphabets
    b =  -1.0/q
    a = 1.0 -(1.0/q)
    if !(cent)
     E = 0.0

     for (e, theta) in H.terms
        edge=Any[]

        [push!(edge, state[j] ) for j in e]

        E += theta[edge...]
     end
    return E
    end

    if cent
        E = 0.0

     for (e, theta) in H.terms
        clrs=Any[]
        r = length(e) #order of interaction
        alphabet_keys =  permutations(Array(1:q), r, asymmetric=true) #No need to generate this everytime

        [push!(clrs, state[j] ) for j in e]
        ct =  Tuple(clrs)

        [E += a^(sum( ct.==c )) * b^(r -  sum(ct.==c)) *theta[c...] for c in alphabet_keys]
     end
    return E
    end

end


function TVD(truth::Dict{}, est::Dict{}, n_samples::Int)
    """
    Total variation distance between two distributions.
    """
    s = 0.0
    for (k ,v) in est
        if haskey(truth, k)
            s+= abs( v -  truth[k])
        else
            s+= v
        end
    end

    for (k,v) in truth
        if !haskey(est, k)
            s+=v
        end
    end





    return s/(2*n_samples)

end



function conditional_energy(u::Int, state::Array{Int64,  1},H::FactorGraph, cent::Bool)
"""
Given a state and a FactorGraph, return its energy
"""
q =  H.n_alphabets
b =  -1.0/q
a = 1.0 -(1.0/q)
if !(cent)
    E = 0.0

    for (e, theta) in H.terms
        if u in e
            edge=Any[]

            [push!(edge, state[j] ) for j in e]

            E += theta[edge...]
        end
    end
    return E
end

if cent
    E = 0.0

    for (e, theta) in H.terms
        if u in e
            clrs=Any[]
            r = length(e) #order of interaction
            alphabet_keys =  permutations(Array(1:q), r, asymmetric=true) #No need to generate this everytime

            [push!(clrs, state[j] ) for j in e]
            ct =  Tuple(clrs)

            [E += a^(sum( ct.==c )) * b^(r -  sum(ct.==c)) *theta[c...] for c in alphabet_keys]
        end
    end
    return E
    end

end


function pth_order_tensor(r)
    a =  Tuple([2 for i in 1:r])
    M =  zeros(a)
    for i in CartesianIndices(M)
        s = prod(2*[Tuple(i)...] .- 3)
        M[i] = s
    end
    return M
end
