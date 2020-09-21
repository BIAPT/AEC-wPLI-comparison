# PLI Generation Comparison
In this first experiment we are assessing why we cannot reproduce the results we had in past milestone when we recalculate the graphs using the wPLI definition. The past graphs were made by Jason Da Silva Castanheira when he was a research assistant at the BIAPT lab.

To work with we had past code made from Dr. Lucrezia Liuzzi and some code that Jason generated. These can be found in the `/legacy/` folder. However, the actual code that was used to generate the matrix isn't directly stated. After some email exchange with Jason :


> The code you have looks like the right one, it adapted the code Lucrezia gave us to the new atlas and loops over all conditions and participants.  It does a bunch of things, including a windowing approach to compute AEC (line 159-168) and PLI (lines 248-253). The output of the code is set up to spit out the average of the windows (AECp), but you can simply change it by outputting the other variables (i.e., for AEC_OUT you  can save the output acep before you take the mean across the windows in (line 233-234) and the same with the PLI (the average PLI across windows is Z_score and the variable with all the windows is PLIcorr; lines 304-312 is a convoluted way to take an average of a 3D Matrix I could have easily done mean(PLICorr,3)). So if you want the windows of AEC and PLI just keep the data before the average and it should be a 3D matrix with the third dimension being window number.


we've realised that they were generated manually. We needed a way of generating them automatically.

The result we were trying to hit is the following in a binary classification between baseline and unconsciouss state & baseline and pre-ROC state.

|      | AEC | PLI |
|------|-----|-----|
| B/U  | 85% | 78% |
| B/PR | 68% | 78% |

### Experiments Attempted
- BIAPT wPLI based graph generation
- Feature generation from old graph
- Lucrezia Liuzzi PLI without surrogates correction
- Jason version (i.e. Lucrezia + wPLI surrogates correction)
- Jason version with modification 1 : surrogates on smaller window
- Jason version with modification 2 : surrogates without mean substraction (**ABORTED**)
- Jason version with modification 3 : surrogates without mean substraction and with smaller window size
- Lucrezia Liuzzi PLI + surrogates
- Lucrezia Liuzzi PLI + surrogates + whole segment (**ABORTED**)

## BIAPT wPLI generation
see: `generate_biapt_wpli.m`

The BIAPT lab usually work with the weighted version of the phase lag index. This is handled by the [NeuroAlgo library](https://github.com/BIAPT/NeuroAlgo). This means that the code snippet looks like this for the windowing version of wPLI over the EEG data:
```matlab
    sampling_rate = fd; % in Hz
    [windowed_data, num_window] = create_sliding_window(filtered_data, window_size, step_size, sampling_rate);

    %% Iterate over each window and calculate pairwise corrected aec
    result = struct();
    pli = zeros(NUM_REGIONS, NUM_REGIONS, num_window);

    parfor win_i = 1:num_window
        disp(strcat("wPLI at window: ",string(win_i)," of ", string(num_window))); 
        
        segment_data = squeeze(windowed_data(win_i,:,:));
        pli(:,:, win_i) = wpli(segment_data', num_surrogates, p_value);
    end
```
This version yield the following result:

|      | AEC | wPLI |
|------|-----|-----|
| B/U  | 85% | 69% |
| B/PR | 68% | 68% |

## Feature generation from old graph
see: `generate_features_from_old_graph.m`

In this analysis we tried to reassure ourselves that everything in the machine learning pipeline was working properly by generating the features.csv matrix from the matrix that Jason generated. We were able to re-generate the exact number we've outlined in the paper. This means that the problem lies in the generation of the (w)PLI graphs.

It yields:

|      | AEC | PLI |
|------|-----|-----|
| B/U  | 85% | 78% |
| B/PR | 68% | 78% |

## Lucrezia Liuzzi PLI (no surrogate)
see: `generate_liuzzi_pli.m`

In this version we used the PLI definition (without the weighted part) which was code originally provided by Dr. Liuzzi. The important part looks like this:

```matlab
ht = hilbert(Vfilt);
    ht = ht(cut+1:end-cut,:);
    ht = bsxfun(@minus,ht,mean(ht,1));
    % Phase information
    theta = angle(ht);

    % Bandwidth
    B = high_frequency-low_frequency;
    % Window duration for PLI calculation
    T = 100/(2*B);                % ~100 effective points
    N = round(T*fd/2)*2;
    K = fix((m-N/2-cut*2)/(N/2)); % number of windows, 50% overlap
    V = nchoosek(R,2);            % number of ROI pairs
    pli_vector = zeros(V,K);

    
    %% Iterate over each window and calculate pairwise corrected aec
    result = struct();
    pli = zeros(NUM_REGIONS, NUM_REGIONS, K);
    
    % Boolean mask to convert to a square matrix
    ind = logical(triu(ones(R),1));
    
    % Loop over time windows
    for k = 1:K

        ibeg = (N/2)*(k-1) + 1;
        iwind = ibeg:ibeg+N-1;

        % loop over all possible ROI pairs
        for jj = 2:R
            ii = 1:jj-1;
            indv = ii + sum(1:jj-2);
            % Phase difference
            RP = bsxfun(@minus,theta(iwind,jj),theta(iwind, ii));
            srp = sin(RP);
            pli_vector(indv,k) = abs(sum(sign(srp),1))/N;

        end
        
        % Attempt at converting from a vector to a pli matrix
        pli_temp = zeros(NUM_REGIONS, NUM_REGIONS);
        pli_temp(ind) = pli_vector(:, k);
        pli_temp = pli_temp + pli_temp';
        
        pli(:, :, k) = pli_temp;
    end
```
This version gives us the following result (remember there is no surrogates analysis hapenning here):

|      | AEC | wPLI |
|------|-----|-----|
| B/U  | 85% | 68% |
| B/PR | 68% | 68% |

## Jason version (Lucrezia + wPLI surrogate correction)
see: `generate_jason_pli.m`

After some reverse engineering and probing, I've understood that the graphs the analysis hinge on were made using the PLI definition of phase lag index + correction using 20 surrogates of **wPLI** matrices. This is important because the surrogates and the actual pli calculation are made with two separate definition of PLI which shouldn't have happened. Furthermore, the surrogates were calculated using the whole recording not the one window that the PLI was calculated on:
```matlab
    ht = hilbert(Vfilt);
    ht = ht(cut+1:end-cut,:);
    ht = bsxfun(@minus,ht,mean(ht,1));
    % Phase information
    theta = angle(ht);

    % Bandwidth
    B = high_frequency-low_frequency;
    % Window duration for PLI calculation
    T = 100/(2*B);                % ~100 effective points
    N = round(T*fd/2)*2;
    K = fix((m-N/2-cut*2)/(N/2)); % number of windows, 50% overlap
    V = nchoosek(R,2);            % number of ROI pairs

    
    %% Iterate over each window and calculate pairwise corrected aec
    result = struct();
    pli = zeros(NUM_REGIONS, NUM_REGIONS, K);
    
    % Boolean mask to convert to a square matrix
    ind = logical(triu(ones(R),1));
    
    % Loop over time windows
    parfor k = 1:K

        ibeg = (N/2)*(k-1) + 1;
        iwind = ibeg:ibeg+N-1;

        % Calculate PLI
        pli_temp =  calculate_pli(theta, iwind, ind, V, R, N)
        
        % Correct it using surrogate analysis (FAULTY VERSION)
        pli(:, :, k) = surrogate_analysis_faulty(Vfilt, pli_temp);
    end
    
    result.wpli = pli;
```
The calculate pli is a bundled version of the PLI calculation from Lucrezia code:
```matlab
function [pli_temp] = calculate_pli(theta, iwind, ind, V, R, N)
%% CALCULATE PLI helper function to wrap the calculation of PLI

        pli_vector = zeros(V,1);
        % loop over all possible ROI pairs
        for jj = 2:R
            ii = 1:jj-1;
            indv = ii + sum(1:jj-2);
            % Phase difference
            RP = bsxfun(@minus,theta(iwind,jj),theta(iwind, ii));
            srp = sin(RP);
            pli_vector(indv) = abs(sum(sign(srp),1))/N;
        end

        % Attempt at converting from a vector to a pli matrix
        pli_temp = zeros(R, R);
        pli_temp(ind) = pli_vector(:);
        pli_temp = pli_temp + pli_temp';
end
```
The surrogate analysis faulty function is the following:
```matlab
function [PLIcorr] = surrogate_analysis_faulty(Vfilt, pli_temp)
%% SURROGATE ANALYSIS FAULTY this is a BAD way of correcting pli
% Two things are wrong here:
% 1) the surrogates are calculated on the whole dataset
% 2) the surrogates are calculated on wPLI while the PLI was calculated
% using the vanilla version
% DO NOT REUSE

%Calculating the surrogate
    display('Calculating surrogate:');
    for j = 1:20
        PLI_surr(j,:,:) = w_PhaseLagIndex_surrogate(Vfilt);
    end

    %Here we compare the calculated dPLI versus the surrogate
    %and test for significance            
    for m = 1:length(pli_temp)
        for n = 1:length(pli_temp)
            test = PLI_surr(:,m,n);
            p = signrank(test, pli_temp(m,n));       
            if p < 0.05
                if pli_temp(m,n) - median(test) < 0 %Special case to make sure no PLI is below 0
                    PLIcorr(m,n) = 0;
                else
                    PLIcorr(m,n) = pli_temp(m,n) - median(test);
                end
            else
                PLIcorr(m,n) = 0;
            end          
        end
    end
            
end
```
finally the w_PhaseLagIndex_surrogate is defined as follow, note that it use the full EEG recording (Vfilt):
```matlab
function surro_WPLI=w_PhaseLagIndex_surrogate(X)
% Given a multivariate data, returns phase lag index matrix
% Modified the mfile of 'phase synchronization'
    ch=size(X,2); % column should be channel
    splice = randi(length(X));  % determines random place in signal where it will be spliced

    a_sig=hilbert(X);
    a_sig2= [a_sig(splice:length(a_sig),:); a_sig(1:splice-1,:)];  % %This is the randomized signal
    surro_WPLI=ones(ch,ch);

    for c1=1:ch-1
        for c2=c1+1:ch
            c_sig=a_sig(:,c1).*conj(a_sig2(:,c2));
            
            numer=abs(mean(imag(c_sig))); % average of imaginary
            denom=mean(abs(imag(c_sig))); % average of abs of imaginary
            
            surro_WPLI(c1,c2)=numer/denom;
            surro_WPLI(c2,c1)=surro_WPLI(c1,c2);
        end
    end 

end
```
The preceding function was used in EEGapp.
The result we get with this is the following:

|      | AEC | wPLI |
|------|-----|-----|
| B/U  | 85% | 81% |
| B/PR | 68% | 76% |

## Jason version first modification
see: `generate_jason_pli_mod_1.m`

We've tried to fix the PLI to add surrogates analysis directly using the pli, but something went wrong. Instead we will incrementally fix Jason code in order to get correct results.

The first improvement being made is to calculate the faulty surrogates only on the window that is being used for analysis (not on the whole EEG recording)

```matlab
    % Correct it using surrogate analysis (FAULTY VERSION)
    ht_win = ht(iwind,:);
    pli(:, :, k) = surrogate_analysis_faulty(ht_win, pli_temp);
```

However, by doing so we see a drastic reduction in the accuracy of PLI:

|      | AEC | wPLI |
|------|-----|-----|
| B/U  | 85% | 69% |
| B/PR | 68% | 58% |

This shows that the part that holds a lot of importance is the accuracy of the surrogate created. However, it might be due by the fact that the hilbert transform has had the mean of the whole recording substracted. Therefore, one other thing to test would be to not subtract the mean while doing the analysis.

## Jason version second modification
see: `generate_jason_pli_mod_2.m`

The second improvement to do is to remove the substraction of the mean hilbert transform and run the analysis as Jason setup with the faulty wPLI with full EEG.

**ABORTED** It took way to long to generate.

## Jason version third modification
see: `generate_jason_pli_mod_3.m`

The third modification is to do the same thing as the second modification, but run it with the first modificaiton. That means that we will still have windowing but not with the mean substracted.


|      | AEC | wPLI |
|------|-----|-----|
| B/U  | 85% | 67% |
| B/PR | 68% | 60% |


## Lucrezia Liuzzi PLI + surrogates
see `genetate_pli_surrogate_graph.m`

In this version we are using the PLI definition without the weighting along with a surrogate analysis of significance using PLI instead of wPLI.

|      | AEC | wPLI |
|------|-----|-----|
| B/U  | 85% | 73% |
| B/PR | 68% | 69% |

This is still lower than what we were expecting, however it is better than the `Jason version first modification` which was faulty + on only small window. Increasing the window for the surrogates to the whole window might help here. Will need to test it out.

## Lucrezia Liuzzi PLI + surrogates + whole segment
see `genetate_pli_surrogate_graph_mod_1.m`

In this version it's the same idea as the previous one **Lucrezia Liuzzi PLI + surrogates** however we will be doing the surrogate analysis of significance on the whole EEG recording everytime.

Here is the result for the analysis:
|      | AEC | wPLI |
|------|-----|-----|
| B/U  | 85% | 72% |
| B/PR | 68% | 68% |

# Conclusion
The general conclusion that we can take for this serie of analysis is that:
- We are able to get comparatively similar values with Jason + surrogates and Lucrezia + surrogates for the small windows. However, they are still not comparable to the result we had previously.
- Running the surrogates correction on the whole recording seems to improve the accuracy for unknown reasons. It might be due that we are creating a much more stable random surrogate (?).
- Using a beefed up setup with multiple nodes and more than 40 cores is **not the way to go**. Surprisingly, it takes way more time to get the ressource than if we just ran the analysis with 1 node and 40 cores. A more modular approach to the codebase would improve the analysis speed because this is too prohibitive in terms of iteration.

The next step for this is to find a way with matlab to generate the result using as much parallele analysis as possible while only using 1 node and 40 cores per analysis. This way we will have higher priority in the cluster.