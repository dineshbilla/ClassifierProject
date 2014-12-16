#! ~/Documents/AI/Proj/WNBayes.m -qf
# Weighted Naive Bayes Classifier
# Based on "Learning Weighted Naive Bayes with Accurate Ranking"
# Computation of weights from gain ratio

X = dlmread("adult_processed.data",',');
start = time();
X = sortrows(X,15);
I = [0,1,0,1,0,1,1,1,1,1,0,0,0,1];
M = max(X);
positive = X( X(:,15)==1,:);
negative = X( X(:,15)==0,:);
meanpositive = mean(positive);
varpositive = var(positive);
meannegative = mean(negative);
varnegative = var(negative);
test = X;
S = size(X,1);
P_pos = size(positive,1)/S;
P_neg = 1-P_pos;
#The mean (mu) and variance(sigma)

#H = entropy_class(X);


#for f=[1:14]
#    if(I(f) == 0)
#      fsplit = cont_entropy(X,f);
#      frac = size(X( X(:,f) <= fsplit(1),:),1)/S;
#      IV(f) = -1*(frac*log(frac) + (1-frac)*log(1-frac));
#    else
#      
#      fsplit = entropy(X,f);
#      IV(f) = 0 ;
#      for u = 1:max(X(:,f))
#        frac = size(X( X(:,f) == u,:),1)/S;
#        IV(f) = IV(f) - frac*log(frac);
#      end
#    endif
#    ent(f) = fsplit(2);
#end

#IG = H - ent;
#GR = IG ./ IV;
#sum_GR = sum(GR);
#WT = GR*14/sum_GR;

# Precomputed weights from the above logic
WT = [1.3196619,0.1741599,0.0079097,0.4607201,1.2423307,1.2436284,0.3943184,1.1168569,0.1538079,0.5913939,4.8149667,1.6649839,0.6540670,0.1611945];

for f =[1:14]
  testf = test(:,f);
  if(I(f) == 0) #cont att
    mu = meanpositive(1,f);
    sigma = varpositive(1,f);
    posprob(:,f) = (1/(sqrt(2*pi)*sigma)*exp(-(testf-mu).^2/(2*sigma)));

  else
    p = positive(:,f);u = 1:M(1,f);
    p = hist(p,u);p = p + 1;
    prob = p/(size(positive(:,f),1)+ M(1,f));
    prob = prob';
    posprob(:,f) = prob(testf,1);
  endif

  posprob(:,f) = posprob(:,f) .^ WT(f);
end

for f =[1:14]
  testf = test(:,f);
  if(I(f) == 0) #cont att
    mu = meannegative(1,f);
    sigma = varnegative(1,f);
    negprob(:,f) = (1/(sqrt(2*pi)*sigma)*exp(-(testf-mu).^2/(2*sigma)));
  else
    p = negative(:,f);u = 1:M(1,f);
    p = hist(p,u);p = p + 1;
    prob = p/(size(negative(:,f),1)+ M(1,f));
    prob = prob';
    negprob(:,f) = prob(testf,1);
  endif
  negprob(:,f) = negprob(:,f) .^ WT(f);
end

P = prod(posprob,2).*P_pos;
N = prod(negprob,2).*P_neg;

result = P > N;

error = result - test(:,15);
error = abs(error);

total_samples = size(X,1)
incorrectly_classified = sum(error)

error_percentage = 100*sum(error)/size(result,1)
endtime = time();
timetaken = endtime - start



