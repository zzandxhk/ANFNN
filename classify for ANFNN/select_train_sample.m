function [train_sample_input, train_sample_output, test_sample_input, test_sample_output, number_train_sample, number_test_sample] = ...
    select_train_sample(input_sample, output_sample)

total_sample = evalin('base', 'total_sample;');
rate_for_train = evalin('base', 'rate_for_train;');

    % introduce the variable in workplace
    index = randperm( total_sample); % ����������
    number_train_sample = round( rate_for_train * total_sample); % ѡ��ѵ����������(��������)
    number_test_sample = total_sample - number_train_sample;     % ѡ��������������
    
    train_sample_input = input_sample(:, index(1: number_train_sample));        %ѡ��ѵ����������
    train_sample_output = output_sample(:, index(1: number_train_sample));      %ѡ��ѵ�������������
    
    test_sample_input = input_sample(:, index( number_train_sample+1 :end));    %ѡ��������������
    test_sample_output = output_sample(:, index( number_train_sample+1 :end));  %ѡ�����������������
end