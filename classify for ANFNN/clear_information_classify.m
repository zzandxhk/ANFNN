function clear_information_classify()
  
evalin('base', 'clear index;')
    evalin('base', 'clear number_train_sample;')
    evalin('base', 'clear number_test_sample;')
    evalin('base', 'clear train_sample_input;')
    evalin('base', 'clear train_sample_output;')
    evalin('base', 'clear number_test_sample;')
    evalin('base', 'clear test_sample_input;')
    evalin('base', 'clear test_sample_output;')
    evalin('base', 'clear number_rules;')
    evalin('base', 'clear a_membership;')
    evalin('base', 'clear width_membership;')
    evalin('base', 'clear link_weights_rule2output;')
    evalin('base', 'clear train_node_input_layer;')
    evalin('base', 'clear train_node_membership_layer;')
    evalin('base', 'clear note_train;')
    evalin('base', 'clear train_node_rule_layer;')
    evalin('base', 'clear train_node_output_layer;')
    evalin('base', 'clear Result_train;')
    evalin('base', 'clear Result_Train_Idea;')
    evalin('base', 'clear test_node_input_layer;')
    evalin('base', 'clear test_node_membership_layer;')
    evalin('base', 'clear note_test;')
    evalin('base', 'clear test_node_rule_layer;')
    evalin('base', 'clear test_node_output_layer;')
    evalin('base', 'clear Result_test;')
    evalin('base', 'clear Result_Test_Idea;')

end