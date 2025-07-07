%保持上一时刻采样值
function [y11,y12,y13,u11,u12,u13]=DMC4(controller)
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
%一共有三个输出，三个微分的时间步
duM11=zeros(N,1);
duM12=zeros(N,1);
duM13=zeros(N,1);
duM=[duM11;duM12;duM13];
%对象模型
step1=[pmodel{1,1};pmodel{1,2};pmodel{1,3}];
step2=[pmodel{2,1};pmodel{2,2};pmodel{2,3}];
step3=[pmodel{3,1};pmodel{3,2};pmodel{3,3}];
%输出序列，对输出序列进行初始化
y1=zeros(N,1);
y2=zeros(N,1);
y3=zeros(N,1);
%初始化输入序列
u=zeros(3,N);

np=controller.np;
V=controller.V;
% 加载数据并赋值
for i = 1:9
    % 生成文件名
    filename = sprintf('y_total%d_%d.mat', np, i);
    
    % 加载数据
    loadedData = load(filename);
    
    % 将加载的数据赋值给 model 对应的单元格
    % 这里假设每个文件中有 y_total 变量
    % 计算行和列的索引
    row = ceil(i / 3); % 计算行索引
    col = mod(i - 1, 3) + 1; % 计算列索引（1 到 3）

    % 将 y_total 转置并赋值给 model
    model{row, col} = loadedData.y_total';
end

controller.model1=model;

%---------------在线计算
%仿真步数
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
