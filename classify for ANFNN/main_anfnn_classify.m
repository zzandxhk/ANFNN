
%    cluster_method == 1: giving the center and width of membership function
%                         with k-means
%    cluster_method == 2: giving the center and width of membership
%                         function with AP algorithm

%    is_nonstationary ==1: with nonstationary process



clc
clear 
close all


load('dataset\Iris.mat')
isNorm = 'y';    % 'n' is not normalized, 'y' is normalized

input = mapminmax(input);
Y=var(input,0,1);
max(Y)
inputEx = mapminmax(inputEx);
clear_useless_information_classify(isNorm);
disp('对于不同的数据集 mat文件，可能命名不同导致报错')

cluster_method = 1;
is_nonstationary = 1;
num_nonstationary = 1;
disturb_coff = [0.5,0.1];    % disturb coefficient

% set the hyperparameter
rate_for_train = 0.5;   % the number of sample used to be train set
MaxEpoches = 1000;      % the maximum number of epoch of the iteration
% learn_rate_GNF = 0.07;  % learning rate 
ap_initial = -10;       % AP initial value

FSglobal_train_acc = zeros(1,10);
FSglobal_test_acc = zeros(1,10);
NFSglobal_train_acc = zeros(1,10);
NFSglobal_test_acc = zeros(1,10);

tic
for i=1:1   
    
%     FSglobal_train_acc(i) = 0;
%     FSglobal_test_acc(i) = 0;
%     NFSglobal_train_acc(i) = 0;
%     NFSglobal_test_acc(i) = 0;
    
    
    for parameter_loop = 1:30
        clear_information_classify();    
        
       
        [train_sample_input, train_sample_output, test_sample_input, test_sample_output, number_train_sample, number_test_sample] = ...
            select_train_sample(input_sample, output_sample);
        

%%%%%%%%%%%%%%%%%↓↓↓↓↓cluster process↓↓↓↓↓%%%%%%%%%%%%%%%%%%%%%
        if(cluster_method == 1)
            %   K-means 
            number_rules = number_classes;   % the number of the fuzzy rules, we set it the number of the classes here;
            [idx, C, a_membership, width_membership] = kmeans_value_parameter(number_rules, number_classes);
            link_weights_rule2output = rand(number_classes,number_rules);
            %   AP
        elseif(cluster_method == 2)
            [C_index,C,idx, a_membership, width_membership, link_weights_rule2output] = ap_value_parameter(number_classes, ap_initial);
            number_rules = length(C_index);             
        end
%%%%%%%%%%%%%%%%%↑↑↑↑↑cluster process↑↑↑↑↑%%%%%%%%%%%%%%%%%%%%%        
 toc       
%%%%%%%%%%%%%%%%%↓↓↓↓↓↓↓↓↓↓%%%%%%%%%%%%%%%%%%%%%        
        learn_rate_GNF = 0.15;  % learning rate
        is_update_memb = 1;
        is_update_width = 1;
        is_update_w = 1;
        [a_membership, width_membership, link_weights_rule2output, Error] = ...
            decent_grad_method_anfnn_classify( a_membership, is_update_memb, width_membership, is_update_width, link_weights_rule2output,is_update_w, 1);
       
        %Robustness test  
%       a_membership = a_membership + normrnd(0, 0.04, size(a_membership,1), size(a_membership,2));
%       width_membership = width_membership + normrnd(0, 0.04, size(width_membership,1), size(width_membership,2));
%       link_weights_rule2output = link_weights_rule2output + normrnd(0, 0.04, size(link_weights_rule2output,1), size(link_weights_rule2output,2));
        
        
        [train_label_output_layer, test_label_output_layer] = print_label_output_layer();
        [currentTrainAcc, currentTestAcc] = print_result_on_dataset_classify(train_label_output_layer, test_label_output_layer);
        
        FSglobal_train_acc(parameter_loop) = currentTrainAcc;
        FSglobal_test_acc(parameter_loop) = currentTestAcc;
        
  
%%%%%%%%%%%%%%%%%↑↑↑↑↑↑↑↑↑↑%%%%%%%%%%%%%%%%%%%%% 
        
        if is_nonstationary == 1 && num_nonstationary >=1
            allModel_train_label = zeros(num_nonstationary,number_train_sample); 
            allModel_test_label = zeros(num_nonstationary,number_test_sample); 
            allModel_train_label(1,:) = train_label_output_layer;
            allModel_test_label(1,:) = test_label_output_layer;
          
            
%%%%%%%%%%%%%%%%%↓↓↓↓↓↓↓↓↓↓%%%%%%%%%%%%%%%%%%%%%                
           
            a_membership_1 = a_membership;   
            width_membership_1 = width_membership;
            link_weights_rule2output_1 = link_weights_rule2output;
            learn_rate_GNF = 0.14;  % learning rate
            is_update_w = 1;
            is_update_memb = 0;
            is_update_width = 1;
            for i_nonstationary = 1:num_nonstationary
                
             
%                 if unifrnd(0,1) < 0.7
%                     is_update_w = 1;
%                     is_update_memb = 0;
%                     is_update_width = 1; 
%                 end
                
                [a_membership, width_membership, link_weights_rule2output, Error] = ...
                    decent_grad_method_anfnn_classify( a_membership_1, is_update_memb, width_membership_1, is_update_width, link_weights_rule2output_1,is_update_w, i_nonstationary);
                
                
%               a_membership = a_membership + normrnd(0, 0.05, size(a_membership,1), size(a_membership,2));
%               width_membership = width_membership + normrnd(0, 0.04, size(width_membership,1), size(width_membership,2));
%               link_weights_rule2output = link_weights_rule2output + normrnd(0, 0.04, size(link_weights_rule2output,1), size(link_weights_rule2output,2));
                

                [train_label_output_layer, test_label_output_layer] = print_label_output_layer();
                allModel_train_label(i_nonstationary,:) = train_label_output_layer;
                allModel_test_label(i_nonstationary,:) = test_label_output_layer;
            end
            
            % 
            Mark = randperm(num_nonstationary);
            Vector_mark = Mark(:,1:num_nonstationary * 0.7);
            allModel_train_label = allModel_train_label(Vector_mark,:);
            
            
            
            train_label_output_layer = mode(allModel_train_label);
            test_label_output_layer = mode(allModel_test_label);
            [currentTrainAcc, currentTestAcc] = print_result_on_dataset_classify(train_label_output_layer, test_label_output_layer);
        end
        
        NFSglobal_train_acc(parameter_loop) = currentTrainAcc;
        NFSglobal_test_acc(parameter_loop) = currentTestAcc;
%%%%%%%%%%%%%%%%%↑↑↑↑↑↑↑↑↑↑%%%%%%%%%%%%%%%%%%%%%   
       
        
    end
    
 
end

result = [FSglobal_train_acc; FSglobal_test_acc; NFSglobal_train_acc; NFSglobal_test_acc]';

% save result_classify.mat
% find(global_test_acc == max(global_test_acc))
% mean(mean(result))
% mean(std(result))
