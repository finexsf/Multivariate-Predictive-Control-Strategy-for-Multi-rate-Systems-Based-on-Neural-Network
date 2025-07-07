%一阶惯性等效模型
function [y11,y12,y13,u11,u12,u13]=DMC8(controller)
model=controller.model1;
pmodel=controller.pmodel;
P=controller.P;%P为预测时域
M=controller.M; %M为控制时域
N=controller.N;%N为模型长度
p=controller.p;  %p为输出个数
m=controller.m;  %m为输入个数
controller.y=[0;0;0];%当前输出
controller.lv=[0.5;-0.5;-0.5];%滤波系数
controller.ym=zeros(p*N,1);%初始预测输出值,对每个输出预测到N步，目标函数计算时使用前P步
controller.A=[];%动态控制矩阵
controller.du=[0;0;0];%输入即时增量
uu=zeros(m,1);
duM11=zeros(N,1);
duM12=zeros(N,1);
duM13=zeros(N,1);
duM=[duM11;duM12;duM13];
step1=[pmodel{1,1};pmodel{1,2};pmodel{1,3}];
step2=[pmodel{2,1};pmodel{2,2};pmodel{2,3}];
step3=[pmodel{3,1};pmodel{3,2};pmodel{3,3}];
%输出序列
y1=zeros(N,1);
y2=zeros(N,1);
y3=zeros(N,1);
y1_1=0;
y2_1=0;
y3_1=0;
u=zeros(3,N);

Ti=controller.Ti;%输入采样周期
To=controller.To;%输出采样周期
np=controller.np;%一个输出采样对应的输入采样数量
V=controller.V;%采样点数/采样时间=采样频率

y1_sample=zeros(V+1,1);
%对非采样点插值
for i=1:3
    for j=1:3
        for ki=1:N
            for k1=1:V
                %取在采样点上的只
                if ki==np*k1
                    kmodel{i,j}(k1,1)=model{i,j}(ki,1);
                end
            end
        end
    end
end

sig = 0; % 核函数宽度
x2=1:1:N;
x1=1:np:(N+np);
for i=1:3
    for j=1:3     
        Y_ALL = Interpolation(kmodel{i,j},np,V);
        model{i,j}=Y_ALL;
    end
end



controller.model1=model;
%---------------在线计算

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