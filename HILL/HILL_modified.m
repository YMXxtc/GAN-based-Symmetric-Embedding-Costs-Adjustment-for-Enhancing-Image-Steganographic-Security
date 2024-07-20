function [stego, prob_map, modify, rhoP1, rhoM1, rhoB] = HILL_modified(cover, payload)

%Get filter
HF=[-1 2 -1; 2 -4 2; -1 2 -1];
H2 = fspecial('average', [3 3]);

% Read image
sizeCover=size(cover);
padsize=max(size(HF));
coverPadded = padarray(cover, [padsize padsize], 'symmetric');% add padding

R = conv2(coverPadded, HF, 'same');%mirror-padded convolution HF和H2卷积操作
% figure(1); imshow(R);
W = conv2(abs(R), H2, 'same');
% figure(2); imshow(W);

% correct the W shift if filter size is even 对偶数卷积核操作后的调整
if mod(size(HF, 1), 2) == 0, W = circshift(W, [1, 0]); end;
if mod(size(HF, 2), 2) == 0, W = circshift(W, [0, 1]); end;

% remove padding
W = W(((size(W, 1)-sizeCover(1))/2)+1:end-((size(W, 1)-sizeCover(1))/2), ((size(W, 2)-sizeCover(2))/2)+1:end-((size(W, 2)-sizeCover(2))/2));


cost=1./(W+10^(-10)); % cost为滤波后的值的倒数
wetCost = 10^10;

% figure(3); imshow(cost);

% compute embedding costs \rho
rhoA = cost;
rhoA(rhoA > wetCost) = wetCost; % threshold on the costs
rhoA(isnan(rhoA)) = wetCost; % if all xi{} are zero threshold the cost

HW =  fspecial('average', [15, 15]) ;
rhoB = imfilter(rhoA, HW ,'symmetric','same'); % HW相关操作

% figure(4); imshow(rhoB);

rhoP1 = rhoB;
rhoM1 = rhoB;

% adjust the rho according to err, +1 if err > 0
% adjustP1 = ones(size(cover)); adjustM1 = ones(size(cover)); 
% temp = abs(ERR(:));  temp = sort(temp); 
% percent = 90;
% alpha = 2;
% idx = ceil(size(temp,1) * percent * 0.01);
% threshold = temp(idx); 
% disp(threshold);
% adjustP1(ERR > threshold) = adjustP1(ERR > threshold) / alpha;  adjustM1(ERR > threshold) = adjustM1(ERR > threshold) * alpha;  
% adjustP1(ERR < -threshold) = adjustP1(ERR < -threshold) * alpha;  adjustM1(ERR < -threshold) = adjustM1(ERR < -threshold) / alpha;  

% rhoP1 = rhoP1 .* adjustP1;
% rhoM1 = rhoM1 .* adjustM1;

rhoP1(rhoP1 > wetCost) = wetCost; rhoM1(rhoM1 > wetCost) = wetCost; 
rhoP1(isnan(rhoP1)) = wetCost; rhoP1(isnan(rhoM1)) = wetCost; 

rhoP1(cover==255) = wetCost; % do not embed +1 if the pixel has max value
rhoM1(cover==0) = wetCost; % do not embed -1 if the pixel has min value
[stego, prob_map, modify] = EmbeddingSimulator(cover, rhoP1, rhoM1, payload*numel(cover), false);


%% --------------------------------------------------------------------------------------------------------------------------
% Embedding simulator simulates the embedding made by the best possible ternary coding method (it embeds on the entropy bound).
% This can be achieved in practice using "Multi-layered  syndrome-trellis codes" (ML STC) that are asymptotically aproaching the bound.
function [y, prob_map, modify] = EmbeddingSimulator(x, rhoP1, rhoM1, m, fixEmbeddingChanges)

    n = numel(x);
    lambda = calc_lambda(rhoP1, rhoM1, m, n);
    pChangeP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    pChangeM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    if fixEmbeddingChanges == 1
        RandStream.setGlobalStream(RandStream('mt19937ar','seed',139187));
    else
        RandStream.setGlobalStream(RandStream('mt19937ar','Seed',sum(100*clock)));
    end
    randChange = rand(size(x));
    y = x;
    y(randChange < pChangeP1) = y(randChange < pChangeP1) + 1;
    y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) = y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) - 1;

    %y = uint8(y);
    prob_map = pChangeP1 + pChangeM1;

    modify = zeros(size(x));
    modify(randChange < pChangeP1) = 1;
    modify(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) = -1;

    function lambda = calc_lambda(rhoP1, rhoM1, message_length, n)

        l3 = 1e+3;
        m3 = double(message_length + 1);
        iterations = 0;
        while m3 > message_length
            l3 = l3 * 2;
            pP1 = (exp(-l3 .* rhoP1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
            pM1 = (exp(-l3 .* rhoM1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
            m3 = ternary_entropyf(pP1, pM1);
            
            disp(m3)
            disp(message_length)
            disp(l3)

            iterations = iterations + 1;
            if (iterations > 10)
                lambda = l3;
                return;
            end
        end

        l1 = 0;
        m1 = double(n);
        lambda = 0;

        alpha = double(message_length)/n;
        % limit search to 30 iterations
        % and require that relative payload embedded is roughly within 1/1000 of the required relative payload
        while  (double(m1-m3)/n > alpha/1000.0 ) && (iterations<30)
            lambda = l1+(l3-l1)/2;
            pP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
            pM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
            m2 = ternary_entropyf(pP1, pM1);
            
            if m2 < message_length
                l3 = lambda;
                m3 = m2;
            else
                l1 = lambda;
                m1 = m2;
            end
            iterations = iterations + 1;
        end
        
    end




    function Ht = ternary_entropyf(pP1, pM1)
        p0 = 1-pP1-pM1;
        P = [p0(:); pP1(:); pM1(:)];
        H = -((P).*log2(P));
        H((P<eps) | (P > 1-eps)) = 0;
        Ht = sum(H);
    end
end
end
