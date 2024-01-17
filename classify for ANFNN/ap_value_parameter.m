function [C_index,C,idx, a_membership, width_membership, link_weights_rule2output] = ap_value_parameter(number_classes, ap_initial)

% get the value of sample form workplace
number_train_sample = evalin('base', 'number_train_sample;');
train_sample_input = evalin('base', 'train_sample_input;');
train_sample_output = evalin('base', 'train_sample_output;');
number_feature = evalin('base', 'number_feature;');

train_sample = [train_sample_input ; train_sample_output]';

% Below is to ues AP Algorithm;
% Arrange as sample * feature;
% idx means the class each sample belongs to
% C means the center node of each classes clustered
fea = train_sample(:, 1:number_feature);
[l,n]=size(fea);

s=zeros([l,l]);

for i=1:l
    for j=1:l
        s(i,j)=-norm(fea(i,:)-fea(j,:));
    end
end



p1=ap_initial * ones(1,l);

%[Num_mid,C,idx] = apcluster(s,p1,fea);
[idx,~,~,~]=apcluster_demo(s,p1);
C_index = unique(idx);
C = train_sample(C_index, :);

link_weights_rule2output = C(:, number_feature+1 :end)';
C = C(:, 1:number_feature);
number_rules = length(C_index);
% Use the cluster center as the center of membership function
a_membership = C';

% Get the value of width from the distance between the center and each node

for k = 1: number_rules
    tempCluster = train_sample(idx== C_index(k),1:number_feature); % ȡ���� K ����������룬��ʱ����
    tempDiff = tempCluster - repmat(C(k,:),size(tempCluster,1),1); %�������� K ����������ֱ��ȥ����ڵ����ģ����ڼ��㵽���ĵ�ľ���
    middle = tempDiff.*tempDiff;    %%% ������㹫ʽ��ƽ��������. middle ��ʾ����ƽ���Ľ����sqrt ��ʾ�����ŵĽ����

    tempdist = sqrt(middle(:,1:number_feature)); %%%% sqrt ��ʾ�����ŵĽ��,����ά�����ô���

    %%%%% ��ÿһ�еı�׼��,��δ洢����
%     tempdist=1./tempdist;
    width_membership(1:number_feature,k) = std(tempdist);  %%%% �� tempdist ��ÿһ�����׼��� k Ϊ����һ��ʱ��
end

end
