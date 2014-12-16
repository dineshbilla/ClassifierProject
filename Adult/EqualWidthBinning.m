#! ~/Documents/AI/Proj/Adult/NB_Discrete/Discretize.m -qf
# Equal Width Binning implementation for Adult Data Set

function Disc(FX,findex)

  I = [0,1,0,1,0,1,1,1,1,1,0,0,0,1];
  I(findex) = 1;
  for k=4:13

    X = FX;
    fmax = max(X(:,findex));
    fmin = min(X(:,findex));
    discrete = fmin:(fmax-fmin)/k:fmax;

    for row =1:size(X,1)
      for j = 1:size(discrete,2)
        if(X(row,findex) <= discrete(1,j))
          X(row,findex) = j;
          break;
        endif
      end
      if(X(row,findex) > k+1)
        X(row,findex) = k+1;
      endif
    end
    test = X;
    M = max(X);

    positive = X( X(:,15)==1,:);
    negative = X( X(:,15)==0,:);
    meanpositive = mean(positive);
    varpositive = var(positive);
    meannegative = mean(negative);
    varnegative = var(negative);

    S = size(X,1);
    P_pos = size(positive,1)/S;
    P_neg = 1-P_pos;

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

#    P = prod(posprob,2);
#    N = prod(negprob,2);

    P = prod(posprob,2).*P_pos;
    N = prod(negprob,2).*P_neg;

    result = P > N;

    error = result - test(:,15);
    error = abs(error);
    total_samples = size(X,1);

    incorrectly_classified(k-3) = sum(error);
    error_percentage(k-3) = 100*sum(error)/size(result,1);

  end
  incorrectly_classified
  error_percentage
  opt_bins = find(incorrectly_classified == min(incorrectly_classified)) + 3

endfunction

FX = dlmread("adult_processed.data",',');

# Call as Disc(X,feature_to_be_discretized)
# Returns the complite profile of classification based on number of bins
