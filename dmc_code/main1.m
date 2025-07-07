clear;
close all;
clc;
delt=0.1;tfinal=120;p=3;m=3;N=tfinal/delt;
%-------------生成阶跃响应系数模型，对象模型为3*3重油分馏塔过程模型（不包含积分过程）
%poly2tfd--内置函数，生成传递函数  tfd2step--内置函数，生成相应的阶跃响应 将阶跃响应的输出存储在元胞model1的各个位置
g11=poly2tfd(4.05,[50 1],0,27);model11=tfd2step(tfinal,delt,1,g11);model1{1,1}=model11(1:N);
g12=poly2tfd(1.77,[60 1],0,28);model12=tfd2step(tfinal,delt,1,g12);model1{1,2}=model12(1:N);
g13=poly2tfd(5.88,[50 1],0,27);model13=tfd2step(tfinal,delt,1,g13);model1{1,3}=model13(1:N);
g21=poly2tfd(5.39,[50 1],0,18);model21=tfd2step(tfinal,delt,1,g21);model1{2,1}=model21(1:N);
g22=poly2tfd(5.72,[60 1],0,14);model22=tfd2step(tfinal,delt,1,g22);model1{2,2}=model22(1:N);
g23=poly2tfd(6.09,[40 1],0,15);model23=tfd2step(tfinal,delt,1,g23);model1{2,3}=model23(1:N);
g31=poly2tfd(4.38,[33 1],0,20);model31=tfd2step(tfinal,delt,1,g31);model1{3,1}=model31(1:N);
g32=poly2tfd(4.42,[44 1],0,22);model32=tfd2step(tfinal,delt,1,g32);model1{3,2}=model32(1:N);
g33=poly2tfd(7.20,[19 1],0,0);model33=tfd2step(tfinal,delt,1,g33);model1{3,3}=model33(1:N);
%---------------------------------------对象模型:仿真/神经网络---------------------------------------------
% 第1路输入对第3路输出的响应增益进行失配,duM左乘这个矩阵,相当于右乘矩阵的转置
% p按输入输出编号
p11=poly2tfd(4.05,[50 1],0,27);pmodel11=tfd2step(tfinal,delt,1,g11);pmodel{1,1}=pmodel11(1:N);
p12=poly2tfd(1.77,[60 1],0,28);pmodel12=tfd2step(tfinal,delt,1,g12);pmodel{1,2}=pmodel12(1:N);
% 增益失配30%
p13=poly2tfd(5.88,[50 1],0,27);pmodel13=tfd2step(tfinal,delt,1,g13);pmodel{1,3}=pmodel13(1:N);
p21=poly2tfd(5.39,[50 1],0,18);pmodel21=tfd2step(tfinal,delt,1,g21);pmodel{2,1}=pmodel21(1:N);
p22=poly2tfd(5.72,[60 1],0,14);pmodel22=tfd2step(tfinal,delt,1,g22);pmodel{2,2}=pmodel22(1:N);
p23=poly2tfd(6.09,[40 1],0,15);pmodel23=tfd2step(tfinal,delt,1,g23);pmodel{2,3}=pmodel23(1:N);
p31=poly2tfd(4.38,[33 1],0,20);pmodel31=tfd2step(tfinal,delt,1,g31);pmodel{3,1}=pmodel31(1:N);
p32=poly2tfd(4.42,[44 1],0,22);pmodel32=tfd2step(tfinal,delt,1,g32);pmodel{3,2}=pmodel32(1:N);
p33=poly2tfd(7.20,[19 1],0,0);pmodel33=tfd2step(tfinal,delt,1,g33);pmodel{3,3}=pmodel33(1:N);

%------------DMC算法参数初始化
% %创建一个结构体，存储模型的各个参数，被控对象是pmodel
steps=100/delt;
controller1.model1=model1;
controller1.P=N;%P为预测长度
controller1.M=steps; %M为控制时域
controller1.N=N;%N为模型长度
controller1.p=3;  %p为输出个数
controller1.m=3;  %m为输入个数
controller1.pmodel=pmodel;
controller1.alpha=0.01;%滤波系数
controller1.ek=0;%初始误差，用于实现误差滤波
%采样周期
controller1.Ti=1;%输入采样周期
controller1.To=20;%输出采样周期
controller1.np=controller1.To/controller1.Ti;%一个输出采样对应的输入采样数量
controller1.V=controller1.N/controller1.To;%采样点数/采样时间=采样频率
%---------------目标设定
controller1.Ys=[0.5;-0.5;-0.5];%输出点设定
%DMC0:保持上一次采样值
[y01,y02,y03,u01,u02,u03]=DMC0(controller1);


%创建一个结构体，存储模型的各个参数，被控对象是pmodel
controller2.model1=model1;
controller2.P=N;%P为预测长度
controller2.M=steps; %M为控制时域
controller2.N=N;%N为模型长度
controller2.p=3;  %p为输出个数
controller2.m=3;  %m为输入个数
controller2.pmodel=pmodel;
controller2.alpha=0.01;%滤波系数
controller2.ek=0;%初始误差，用于实现误差滤波
%采样周期
controller2.Ti=1;%输入采样周期
controller2.To=20;%输出采样周期
controller2.np=controller2.To/controller2.Ti;%一个输出采样对应的输入采样数量
controller2.V=controller2.N/controller2.To;%采样点数/采样时间=采样频率
% DMC3:二次样条插值
controller2.Ys=[0.5;-0.5;-0.5];%输出点设定
[y21,y22,y23,u21,u22,u23]=DMC3(controller2);


% %创建一个结构体，存储模型的各个参数，被控对象是pmodel
% controller3.model1=model1;
% controller3.P=N;%P为预测长度
% controller3.M=steps; %M为控制时域
% controller3.N=N;%N为模型长度
% controller3.p=3;  %p为输出个数
% controller3.m=3;  %m为输入个数
% controller3.pmodel=pmodel;
% controller3.alpha=0.01;%滤波系数
% controller3.ek=0;%初始误差，用于实现误差滤波
% %采样周期
% controller3.Ti=1;%输入采样周期
% controller3.To=600;%输出采样周期
% controller3.np=controller3.To/controller3.Ti;%一个输出采样对应的输入采样数量
% controller3.V=controller3.N/controller3.To;%采样点数/采样时间=采样频率
% % DMC2:一阶惯性系统插值
% controller3.Ys=[0.5;-0.5;-0.5];%输出点设定
% [y31,y32,y33,u31,u32,u33]=DMC2(controller3);


%创建一个结构体，存储模型的各个参数，被控对象是pmodel
controller4.model1=model1;
controller4.P=N;%P为预测长度
controller4.M=steps; %M为控制时域
controller4.N=N;%N为模型长度
controller4.p=3;  %p为输出个数
controller4.m=3;  %m为输入个数
controller4.pmodel=pmodel;
controller4.alpha=0.01;%滤波系数
controller4.ek=0;%初始误差，用于实现误差滤波
%采样周期
controller4.Ti=1;%输入采样周期
controller4.To=20;%输出采样周期
controller4.np=controller4.To/controller4.Ti;%一个输出采样对应的输入采样数量
controller4.V=controller4.N/controller4.To;%采样点数/采样时间=采样频率
% DMC4:神经网络插值
controller4.Ys=[0.5;-0.5;-0.5];%输出点设定
[y41,y42,y43,u41,u42,u43]=DMC4(controller4);


figure(1);
plot(y01,'-blue');
hold on;
plot(y21,'-red');
hold on;
plot(y41,'-green');
hold on;
plot(0.5*ones(1,steps),'-.');
hold on;
ylabel('y1 error');
xlabel('t/min');
legend('保持上一时刻采样值不变','二次样条插值','神经网络插值','参考轨迹','location','best');
title('T=20T0')

figure(2);
plot(y02,'-blue');
hold on;
plot(y22,'-red');
hold on;
plot(y42,'-green');
hold on;
plot(-0.5*ones(1,steps),'-.');
hold on;
ylabel('y2 error');
xlabel('t/min');
legend('保持上一时刻采样值不变','二次样条插值','神经网络插值','参考轨迹','location','best');
title('T=20T0')

figure(3);
plot(y03,'-blue');
hold on;
plot(y23,'-red');
hold on;
plot(y43,'-green');
hold on;
plot(-0.5*ones(1,steps),'-.');
hold on;
ylabel('y3 error');
xlabel('t/min');
legend('保持上一时刻采样值不变','二次样条插值','神经网络插值','参考轨迹','location','best');
title('T=20T0')
    
    
