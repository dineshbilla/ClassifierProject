#! ~/Documents/AI/Proj/Nbayes.m -qf
#Naive Bayes Classifier

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

# error reported = 18.881
# time =0.47321


