module AdvDT

using Base.Threads
const MAX_DEPTH = 10

mutable struct Node # not sure when Node{T} is used 20210722, alternatively children::Array{Node, 1}
    impurity::Float64 # impurity of this node
    ŷ::Int64 # predicted class
    ι::Int64 # index of the field to split
    θ::Float64 # threshold in the field to determine which path to take
    λ::Int64 # level of the node
    left::Union{Nothing, Node} # left child
    right::Union{Nothing, Node} # right child
    Node(p, y, i, t, l) = new(p, y, i, t, l, nothing, nothing)
end

"""
"""
function gini_impurity_and_target_class(y)
    κ = length(Set(y)) # number of classes
    n = length(y) # number of records
    ηs_cn = [sum(y .== i) for i in 1:κ] # number of samples per class in current node
    return 1.0 - sum((η / n) * (η / n) for η in ηs_cn), argmax(ηs_cn) # most probable class
end

"""
"""
function best_split(X, y, Κ)
    n, m = size(X)
    @assert n == length(y)

    ι⭑, θ⭑ = -1, 0.0
    if n <= 1
        return ι⭑, θ⭑
    end

    ηs_cn = [sum(y .== i) for i in 1:Κ] # number of samples per class in current node
    𝐺⭑ = 1.0 - sum((η / n) * (η / n) for η in ηs_cn)

    for mᵢ in 1:m
        sub = hcat(X[:, mᵢ], y)
        sorted_sub = sub[sortperm(sub[:, 1]), :] # θs = sorted_sub[:, 1], κs = sorted_sub[:, 2]
        ηₗ = zeros(Κ) # can come with a subscript but not a big special character
        ηᵣ = copy(ηs_cn) # can come with a subscript but not a big special character
        for nᵢ in 2:n # default sort order is from small to large
            κᵢ = sorted_sub[nᵢ - 1, 2] # corresponding class
            ηₗ[κᵢ] += 1 # more points towards left
            ηᵣ[κᵢ] -= 1 # fewer points towards right
            if sorted_sub[nᵢ, 1] == sorted_sub[nᵢ - 1, 1] # if the same feature value then no difference
                continue
            end
            𝐺ₗ = 1.0 - sum((ηₗ[κ] / nᵢ) * (ηₗ[κ] / nᵢ) for κ in 1:Κ)
            𝐺ᵣ = 1.0 - sum((ηᵣ[κ] / (n - nᵢ)) * (ηᵣ[κ] / (n - nᵢ)) for κ in 1:Κ)
            𝐺 = (nᵢ * 𝐺ₗ + (n - nᵢ) * 𝐺ᵣ) / n
            if 𝐺 < 𝐺⭑
                𝐺⭑ = 𝐺
                ι⭑ = mᵢ
                θ⭑ = (sorted_sub[nᵢ, 1] + sorted_sub[nᵢ - 1, 1]) / 2
            end
        end
    end

    return ι⭑, θ⭑
end

"""
"""
function best_split_thread(X, y, Κ)
    n, m = size(X)
    @assert n == length(y)

    ι⭑, θ⭑ = -1, 0.0
    if n <= 1
        return ι⭑, θ⭑
    end

    ηs_cn = [sum(y .== i) for i in 1:Κ] # number of samples per class in current node
    𝐺⭑ = 1.0 - sum((η / n) * (η / n) for η in ηs_cn)

    @threads for mᵢ in 1:m
        sub = hcat(X[:, mᵢ], y)
        sorted_sub = sub[sortperm(sub[:, 1]), :] # θs = sorted_sub[:, 1], κs = sorted_sub[:, 2]
        ηₗ = zeros(Κ) # can come with a subscript but not a big special character
        ηᵣ = copy(ηs_cn) # can come with a subscript but not a big special character
        for nᵢ in 2:n # default sort order is from small to large
            if sorted_sub[nᵢ, 1] == sorted_sub[nᵢ - 1, 1] # if the same feature value then no difference
                continue
            end
            κᵢ = sorted_sub[nᵢ - 1, 2] # corresponding class
            ηₗ[κᵢ] += 1 # more points towards left
            ηᵣ[κᵢ] -= 1 # fewer points towards right
            𝐺ₗ = 1.0 - sum((ηₗ[κ] / nᵢ) * (ηₗ[κ] / nᵢ) for κ in 1:Κ)
            𝐺ᵣ = 1.0 - sum((ηᵣ[κ] / (n - nᵢ)) * (ηᵣ[κ] / (n - nᵢ)) for κ in 1:Κ)
            𝐺 = (nᵢ * 𝐺ₗ + (n - nᵢ) * 𝐺ᵣ) / n
            if 𝐺 < 𝐺⭑
                𝐺⭑ = 𝐺
                ι⭑ = mᵢ
                θ⭑ = (sorted_sub[nᵢ, 1] + sorted_sub[nᵢ - 1, 1]) / 2
            end
        end
    end

    return ι⭑, θ⭑
end

"""
Build a decision tree classifier
"""
function fit(X, y, method = "recursive")
    Κ = length(Set(y)) # total number of classes, assuming all integers

    if method == "recursive"
        tree = grow_tree_recursive(X, y, Κ)
    elseif method == "iterative"
        tree = grow_tree_iterative(X, y)
    elseif method == "threaded"
        tree = grow_tree_iterative_thread(X, y)
    else
        println(method * " not supported")
    end

    #=for r in 1:size(X)[1]
        println(predict(tree, X[r, :]))
    end=#
    return tree
end

"""
"""
function predict(tree, record)
    node = tree
    while !isnothing(node.left)
        if record[node.ι] < node.θ
            node = node.left # node.left may be #undef
        else
            node = node.right # node.right may be #undef
        end
    end
    return node.ŷ
end

"""
"""
function print_tree(tree)
    node = tree
    println(node.ŷ, ' ', node.ι, ' ', node.θ, ' ', node.λ)
    if !isnothing(node.left)
        print_tree(node.left)
    end
    if !isnothing(node.right)
        print_tree(node.right)
    end
end

"""
Build a decision tree by recursively finding the best split.
"""
function grow_tree_recursive(X, y, Κ, depth = 0)
    𝑔, ŷ = gini_impurity_and_target_class(y)
    ν = Node(𝑔, ŷ, -1, 0.0, depth) # information in current node
    
    if depth < MAX_DEPTH
        ι⭑, θ⭑ = best_split(X, y, Κ)
        if ι⭑ > -1
            ιsₗ = X[:, ι⭑] .< θ⭑ # indices corresponding to left
            ιsᵣ = X[:, ι⭑] .>= θ⭑ # indices corresponding to right
            Xₗ, yₗ = X[ιsₗ, :], y[ιsₗ]
            Xᵣ, yᵣ = X[ιsᵣ, :], y[ιsᵣ]
            ν.ι = ι⭑ # best field index
            ν.θ = θ⭑ # best field value threshold
            ν.left = grow_tree_recursive(Xₗ, yₗ, Κ, depth + 1)
            ν.right = grow_tree_recursive(Xᵣ, yᵣ, Κ, depth + 1)
        end
    end

    return ν
end

"""
Build a decision tree by iteratively finding the best split.
Many trees and forests can be parallelized.
"""
function grow_tree_iterative(X, y)
    Κ = length(Set(y))
    ν = Node(0.0, 0, -1, 0.0, 0) # information in current node

    node_stack = []
    push!(node_stack, (ν, Bool.(ones(1:size(X)[1])))) # all records
    while length(node_stack) > 0
        (γ, ιs) = pop!(node_stack)
        γ.impurity, γ.ŷ = gini_impurity_and_target_class(y[ιs])
        if γ.λ < MAX_DEPTH
            ι⭑, θ⭑ = best_split(X[ιs, :], y[ιs], Κ)
            if ι⭑ > -1
                γ.ι = ι⭑ # best field index
                γ.θ = θ⭑ # best field value threshold
                γ.left = Node(0.0, 0, -1, 0.0, γ.λ + 1) # information in left child
                γ.right = Node(0.0, 0, -1, 0.0, γ.λ + 1) # information in right child
                push!(node_stack, (γ.left, (ιs) .& (X[:, ι⭑] .< θ⭑)))
                push!(node_stack, (γ.right, (ιs) .& (X[:, ι⭑] .>= θ⭑)))
            end
        end        
    end

    return ν
end

"""
Build a decision tree by iteratively finding the best split using threads.
"""
function grow_tree_iterative_thread(X, y)
    Κ = length(Set(y))
    ν = Node(0.0, 0, -1, 0.0, 0) # information in current node

    node_stack = []
    push!(node_stack, (ν, Bool.(ones(1:size(X)[1])))) # all records
    #=while length(node_stack) > 0
        (ν, ιs) = pop!(node_stack)
        ι⭑, θ⭑ = best_split_thread(X[ιs, :], y[ιs], Κ)
        if ι⭑ > -1
            ν.ι = ι⭑ # best field index
            ν.θ = θ⭑ # best field value threshold
            push!(node_stack, (ν.left, X[ιs, ι⭑] .< θ⭑))
            push!(node_stack, (ν.right, X[ιs, ι⭑] .>= θ⭑))
        end        
    end=#

    return ν
end

end # module
