%������һʱ�̲���ֵ
function [y11,y12,y13,u11,u12,u13]=DMC0(controller)
model=controller.model1;
pmodel=controller.pmodel;
P=controller.P;%PΪԤ��ʱ��
M=controller.M; %MΪ����ʱ��
N=controller.N;%NΪģ�ͳ���
p=controller.p;  %pΪ�������
m=controller.m;  %mΪ�������
controller.y=[0;0;0];%��ǰ���
controller.lv=[0.5;-0.5;-0.5];%�˲�ϵ��
controller.ym=zeros(p*N,1);%��ʼԤ�����ֵ,��ÿ�����Ԥ�⵽N����Ŀ�꺯������ʱʹ��ǰP��
controller.A=[];%��̬���ƾ���
controller.du=[0;0;0];%���뼴ʱ����
uu=zeros(m,1);
%һ�����������������΢�ֵ�ʱ�䲽
duM11=zeros(N,1);
duM12=zeros(N,1);
duM13=zeros(N,1);
duM=[duM11;duM12;duM13];
%����ģ��
step1=[pmodel{1,1};pmodel{1,2};pmodel{1,3}];
step2=[pmodel{2,1};pmodel{2,2};pmodel{2,3}];
step3=[pmodel{3,1};pmodel{3,2};pmodel{3,3}];
%������У���������н��г�ʼ��
y1=zeros(N,1);
y2=zeros(N,1);
y3=zeros(N,1);
%��ʼ����������
u=zeros(3,N);

np=controller.np;
V=controller.V;

%�Էǲ������ֵ
for i=1:3
    for j=1:3
        for ki=1:N
            for k1=1:(V-1)
                %����պô�������������϶�ȡ���ֵ�����������ά��ǰһʱ�̵Ĳ���ֵ����
                if ki==np*k1||ki==1
                    model{i,j}(ki,1)=model{i,j}(ki,1);
                    for x=ki:ki+np-1
                       model{i,j}(x,1)=model{i,j}(ki,1);
                    end
                end
            end
        end
    end
end
controller.model1=model;

%---------------���߼���
%���沽��
for k=1:M
    controller.k=k;
    controller=DMCfun0(controller);
    du=controller.du;
    uu=uu+du;
    u(:,k)=uu;
    duM11=[du(1);duM11(1:N-1,1)];
    duM12=[du(2);duM12(1:N-1,1)];
    duM13=[du(3);duM13(1:N-1,1)];
    duM=[duM11;duM12;duM13];

    y1_1=step1'*duM;
    y2_1=step2'*duM;
    y3_1=step3'*duM;
    controller.y=[y1_1;y2_1;y3_1];
    y1(k)=y1_1;
    y2(k)=y2_1;
    y3(k)=y3_1;
    save('data0.mat','y1','y2','y3','u');
end

y11=y1(1:M,1);y12=y2(1:M,1);y13=y3(1:M,1);
u11=u(1,1:M);u12=u(2,1:M);u13=u(3,1:M);
end
