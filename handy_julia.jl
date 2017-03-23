"""
findin_index(x,y) returns a vector v of the same size as that of x
- where v[i] = index of the element x[i] in the vector y
- v[i] = 0 if x[i] does not exist in y
- assumption: If y does not consist of unique elements, the index returned is the last occurence
```
example:
    julia> x = [1,2,3,10,1,4];
    julia> y = [1,2,5,4,3];
    julia> findin_index(x,y)
    6-element Array{Int64,1}:
     1
     2
     5
     0
     1
     4
```
"""
function findin_index{T}(x::Vector{T},y::Vector{T})
  indices_in_y = zeros(Int64,length(x))
  already_exist = findin(x,y)
  donot_exist = setdiff(1:length(x),already_exist)
  indices_in_y[donot_exist] = 0
  funcmap = i -> indices_in_y[find(x.==y[i])] = i
  lookfor_indices = findin(y,x)
  map(funcmap,lookfor_indices)
  return indices_in_y
end
"""
split_train_test(A,t) returns the matrics Atrain and Atest where:
- Atrain+Atest = A
- nnz(Atrain) is almost equal to t percent
```
example:
    julia> A = sprand(1000,1000,0.3);
    julia> Atrain,Atest = split_train_test(A,0.9);
    julia> nnz(Atrain)/nnz(A)
      0.8999973389393012
    julia> isequal(Atrain+Atest,A)
      true
```
"""
function split_train_test{T}(R::SparseMatrixCSC{T,Int64},rho::Float64)
if !(0<=rho<=1)
  error("function split_train_test: rho must be between 0 and 1. Read `? split_train_test` for more")
end

m,n = size(R)
ei,ej,ev = findnz(R)
len = length(ev)
seed = time()
r = MersenneTwister(round(Int64,seed))
a = randperm(r,len)
nz = floor(Int,rho*len)
p = a[1:nz]
cp = setdiff(collect(1:len),p);

Rtrain = sparse(ei[p],ej[p],ev[p],m,n)
Rtest = sparse(ei[cp],ej[cp],ev[cp],m,n)
return(Rtrain,Rtest)
end
"""
writeToSMAT writes a matrix to a file in SMAT format
```
example:
    julia> A = rand(4,4)
    julia> writeToSMAT(A,"testfile.smat")
```
"""
function writeToSMAT(A,filename)
  i,j,v = findnz(A)
  M = [i';j';v']
  M[2,:] = M[2,:]-1
  M[1,:] = M[1,:]-1
  m = size(A,1)
  n = size(A,2)
  nz = length(v)
  open(filename, "w") do f
    write(f, "$m   $n   $nz \n")
    writedlm(f,M')
  end
end

"""
ismember(A,x,2) returns the indices of rows in A that are equal to x
ismember(A,x,1) returns the indices of cols in A that are equal to x
```
example:
    julia> A = [1 2 3;1 2 4; 2 4 4]
    julia> ismember(A,[2,4,4],2)
          # false
          # false
          # true
    julia> ismember(A,[2,4,4],1)
          # false  false  false
    julia> ismember(A,[2,2,4],1)
          # false  true  false
```
"""
function ismember{T}(A::Array{T,2},x::Vector{T},dims::Int)
  assert(dims==1 || dims==2)
  # dims = 1 means we're looking at columns
  # dims = 2 means we're looking at rows
  if dims == 2
    sz = size(A,1)
    A = A'
  else
    sz = size(A,2)
  end
  ret = trues(sz)
  map(i->ret[i] = A[:,i] == vec(x),1:sz)
  return ret
end

"""
sortcolsperm(A,true) returns the indices of sorted columns in A in descending order
sortcolsperm(A,false) returns the indices of sorted columns in A in ascending order
```
example:
  julia> W = rand(3,3)
    3×3 Array{Float64,2}:
    0.661943  0.00517749  0.332394
    0.716344  0.61179     0.544258
    0.372336  0.994069    0.297704

  julia> sortcolsperm(W,true)
    3×3 Array{Int64,2}:
    2  3  2
    1  2  1
    3  1  3

  julia> sortcolsperm(W,false)
    3×3 Array{Int64,2}:
    3  1  3
    1  2  1
    2  3  2
```
"""
function sortcolsperm{T}(X::Matrix{T},REV::Bool)
    P = Matrix{Int}(size(X,1),size(X,2))

    Threads.@threads for i=1:size(X,2)
        P[:,i] = sortperm(X[:,i]; rev=REV)
    end
    return P
end
