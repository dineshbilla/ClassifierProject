#! ~/Documents/AI/Proj/Adult/DTree.m -qf
# Decision Tree Classsifier
# Refers the Adult Dataset from the UCI repository

X = dlmread("adult_processed.data",',');
X = sortrows(X,15);
I = [0,7,0,16,0,7,14,6,5,2,0,0,0,41];
F = 1:14;

positive = X( X(:,15)==1,:);
negative = X( X(:,15)==0,:);

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
  if(size(u,1) == 1)
    split(1) = u; split(2) = entropy_class(Y);
    return;
  endif

  #array to store split and entropy
  split_ent = zeros(size(u,1)-1,2);

  for i=1:(size(u,1)-1)
    part = u(i);ctype = Y; 
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

  if(size(split,1) > 1)
    split = split(1,:);
  endif

endfunction

# Measures entropy of discrete attributes
function split = entropy (Y,index)
  I = [0,7,0,16,0,7,14,6,5,2,0,0,0,41];
  split = zeros(1,2);

  types = max(Y(:,index));values = 1:types; 
  ent = 0;
  for i=1:types
    ctype = Y( Y(:,index) == values(1,i),:);
    ent = ent + (size(ctype,1)/size(Y,1))*entropy_class(ctype);
  end
  split(1,2) = ent;
endfunction

# ret indicates [error, number of leaves]
function ret = BuildTree(X,F,height)
  I = [0,7,0,16,0,7,14,6,5,2,0,0,0,41];
  skipF = [3,11];
  ret = [0,0]; # track the error 

  if(size(F,2) == (2) || # leaf node, count the errors
    height == 3) # stop if height is more than 3
    positive = X( X(:,15)==1,:); positive = size(positive,1);
    negative = X( X(:,15)==0,:); negative = size(negative,1);
    err = min([positive,negative]);
    ret = [err,1];
    return;
  endif
   
  if(entropy_class(X) == 0) # perfect split ,stop here
    ret = [0,1];
    return;
  endif

  split_ent = zeros(size(F,2),2);
  i=1;

  for feature = F
    if(feature == 3 || feature == 11)
      split_ent(i,2) = 100;
      i = i + 1;
      continue;
    endif
    if(I(feature) == 0)
      split_ent(i,:) =  cont_entropy(X,feature);
    else
      split_ent(i,:) =  entropy(X,feature);
    endif
    i = i+1;
  end
  findex = find(split_ent(:,2) == min(split_ent(:,2)));
  
  if(size(findex,1) > 1)
    findex = findex(2);
  endif

  fsplit = F(findex);
  F(findex) = []; # delete from the list of features

  # split along that feature
  if(split_ent(findex,1) == 0)
    
    types = I(fsplit); 
    for i=1:types
      ctype = X( X(:,fsplit) == i,:);
      if(size(ctype) > 0)
        ret = ret + BuildTree(ctype,F,height+1);
      endif
    end

  else
    ctype = X( X(:,fsplit) <= split_ent(findex,1),:);
    if(size(ctype) > 0)
      ret = ret + BuildTree(ctype,F,height+1);
    endif

    ctype = X( X(:,fsplit) > split_ent(findex,1),:);
    if(size(ctype) > 0)         
      ret = ret + BuildTree(ctype,F,height+1);
    endif

  endif

endfunction

start = time();
total_samples = size(X,1)
ret = BuildTree(X,F,0);
leaves = ret(2)
incorrectly_classified = ret(1) 
error_percentage = 100*(incorrectly_classified/size(X,1))
endtime = time();
timetaken = endtime - start


