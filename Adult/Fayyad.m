#! ~/Documents/AI/Proj/Adult/Fayyad/Fayyad.m -qf
# Simple implementation of discretization by Fayyad and Irani
# Multi Interval Discretization of Continous Valued Attributes for Classification Learning
# http://ijcai.org/Past%20Proceedings/IJCAI-93-VOL2/PDF/022.pdf


# X is the data set
# F is feature to discretize
function result = Fayyad(X,F)

# first get the entropy of class, if this is zero, stop as we have
# a perfect split, else go ahead and try to split
  result = [];
  if(entropy_class(X) == 0) # perfect split ,stop here
    return;
  endif

  old_ent = entropy_class(X);
  new_ent = cont_entropy(X,F);
# new_ent(1)

  gain = old_ent - new_ent(2);

# Cutoff gain to decide on split
# gain >= (1/N) x log2(N-1)  + (1/N) x [ log2 (3k-2) - (k x Entropy(S) – k1 x Entropy(S1) – k2 x Entropy(S2) ]

  N = size(X,1); k = size(unique(X(:,15)),1);
  left = X( X(:,F) <= new_ent(1),:);
  right = X( X(:,F) > new_ent(1),:);
  
  k1 = size(unique(left(:,15)),1);
  k2 = size(unique(right(:,15)),1);
  
  mingain = log2(N-1) + (log2(3**k-2) - k*entropy_class(X) -k1*entropy_class(left) - k2*entropy_class(right));
  mingain = mingain/N;
  
  
  if(gain > mingain) # split 
    result = [result new_ent(1)];
    if(size(left) > 0)
      r1 = Fayyad(left,F);
      result = [result r1];
    endif
    if(size(right) > 0)         
      r2 = Fayyad(right,F);
      result = [result r2];
    endif
  endif
endfunction


function e = entropy_class(Z)

  p = Z( Z(:,15)==1,:); p = p(:,15); p= size(p,1);
  n = Z( Z(:,15)==0,:); n = n(:,15); n= size(n,1);

  if (p == 0)
    e = 0;
  elseif (n == 0)
    e = 0;
  else
    s = p+n;p=p/s;n=n/s;
    e = p*log(p) + n*log(n); e=e*-1;
  endif

endfunction

# Measures entropy of continous attributes
function split = cont_entropy(Y,index)

  u = unique(Y(:,index));
#  if(index == 1)
#    u'
#  endif

  if(size(u,1) == 1)
    split(1) = u; split(2) = entropy_class(Y);
    return;
  endif


  #array to store split and entropy
  split_ent = zeros(size(u,1)-1,2);
#  if(size(Y,1) == 10)
#    u'
#    Y
#    split_ent
#  endif

  for i=1:(size(u,1)-1)
    part = u(i);ctype = Y; 
#    if(size(Y,1) == 10)
#      part
#    endif
    #ctype = Y( Y(:,index) <= part,:);
    ctype(:,index) = (ctype(:,index) > part)+1;
    
    ent = 0;

    for j=1:2
      cc = ctype( ctype(:,index) == j,:);
      ent = ent + (size(cc,1)/size(ctype,1))*entropy_class(cc);
    split_ent(i,1) = part;
    end
    split_ent(i,2) = ent;
  end
  
  split  = split_ent(find(split_ent(:,2) == min(split_ent(:,2))),:);
#  if(size(Y,1) == 10)
#    split
#  endif

# If there are multiple attributes along which the split can be made
# choose the first one
  if(size(split,1) > 1)
    split = split(1,:);
  endif

endfunction


function R =  Discretize(X,index)
  
#  X = dlmread("adult_processed.data",',');
  result = Fayyad(X,index);
  result = sort(result);
  
  for row =1:size(X,1)
    for j = 1:size(result,2);
      if(X(row,index) <= result(j))
        X(row,index) = j;
        break;
      endif
    end
    if(X(row,index) > j+1)
      X(row,index) = j+1;
    endif
  end
  
  R = X;

endfunction

X = dlmread("adult_processed.data",',');
X = Discretize(X,1);
# 27   23   21   35   29   61   43
X = Discretize(X,5);
# 12    8   10    9   13   14
X = Discretize(X,12);
# 1816   1564   1539   1977   1876   1848   1974   1902   2206   2149   2174   2377   2258   2559   2754   3004
X = Discretize(X,13);
# 41   34   39   49

dlmwrite("adult_discrete.data",X,',');
#save adult_discrete.data X

## total_samples =  30162
## incorrectly_classified =  5017
## error_percentage =  16.634
## timetaken =  0.046004



