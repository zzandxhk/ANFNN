function  [idx, C, a_membership, width_membership] = kmeans_value_parameter(number_rules, number_classes)
    
    % get the value of sample form workplace
    train_sample_input = evalin('base', 'train_sample_input;');
    number_feature = evalin('base', 'number_feature;');
    
    train_sample = [train_sample_input]';

    % Below is to ues algorithm K-means; 
    % Arrange as sample * feature;
    % idx means the class each sample belongs to 
    % C means the center node of each classes clustered
    [idx, C] = kmeans(train_sample, number_rules, 'dist','sqEuclidean','rep',4);   
    
    
    % Use the cluster center as the center of membership function
    a_membership = C';     
    % Get the value of width from the distance between the center and each node
    
    for k = 1: number_rules
        tempCluster = train_sample(idx==k,1:number_feature); % ȡ���� K ����������룬��ʱ����
        tempDiff = tempCluster - repmat(C(k,:),size(tempCluster,1),1); %�������� K ����������ֱ��ȥ����ڵ����ģ����ڼ��㵽���ĵ�ľ���
        middle = tempDiff.*tempDiff;    %%% ������㹫ʽ��ƽ��������. middle ��ʾ����ƽ���Ľ����sqrt ��ʾ�����ŵĽ����
        
        tempdist = sqrt(middle); %%%% sqrt ��ʾ�����ŵĽ��,����ά�����ô���
        
        %%%%% ��ÿһ�еı�׼��,��δ洢����
%         tempdist=1./tempdist;
        width_membership(:,k) = std(tempdist);  %%%% �� tempdist ��ÿһ�����׼��� k Ϊ����һ��ʱ��
    end
   
end
    