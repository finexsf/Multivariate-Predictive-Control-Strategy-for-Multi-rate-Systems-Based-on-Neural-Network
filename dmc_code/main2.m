clear;
close all;
clc;
delt=0.1;tfinal=120;p=3;m=3;N=tfinal/delt;
%-------------���ɽ�Ծ��Ӧϵ��ģ�ͣ�����ģ��Ϊ3*3���ͷ���������ģ�ͣ����������ֹ��̣�
%poly2tfd--���ú��������ɴ��ݺ���  tfd2step--���ú�����������Ӧ�Ľ�Ծ��Ӧ ����Ծ��Ӧ������洢��Ԫ��model1�ĸ���λ��
g11=poly2tfd(4.05,[50 1],0,27);model11=tfd2step(tfinal,delt,1,g11);model1{1,1}=model11(1:N);
g12=poly2tfd(1.77,[60 1],0,28);model12=tfd2step(tfinal,delt,1,g12);model1{1,2}=model12(1:N);
g13=poly2tfd(5.88,[50 1],0,27);model13=tfd2step(tfinal,delt,1,g13);model1{1,3}=model13(1:N);
g21=poly2tfd(5.39,[50 1],0,18);model21=tfd2step(tfinal,delt,1,g21);model1{2,1}=model21(1:N);
g22=poly2tfd(5.72,[60 1],0,14);model22=tfd2step(tfinal,delt,1,g22);model1{2,2}=model22(1:N);
g23=poly2tfd(6.09,[40 1],0,15);model23=tfd2step(tfinal,delt,1,g23);model1{2,3}=model23(1:N);
g31=poly2tfd(4.38,[33 1],0,20);model31=tfd2step(tfinal,delt,1,g31);model1{3,1}=model31(1:N);
g32=poly2tfd(4.42,[44 1],0,22);model32=tfd2step(tfinal,delt,1,g32);model1{3,2}=model32(1:N);
g33=poly2tfd(7.20,[19 1],0,0);model33=tfd2step(tfinal,delt,1,g33);model1{3,3}=model33(1:N);
%---------------------------------------����ģ��:����/������---------------------------------------------
% ��1·����Ե�3·�������Ӧ�������ʧ��,duM����������,�൱���ҳ˾����ת��
% pij������������
p11=poly2tfd(4.05,[50 1],0,27);pmodel11=tfd2step(tfinal,delt,1,g11);pmodel{1,1}=pmodel11(1:N);
p12=poly2tfd(1.77,[60 1],0,28);pmodel12=tfd2step(tfinal,delt,1,g12);pmodel{1,2}=pmodel12(1:N);
% ����ʧ��30%
p13=poly2tfd(5.88,[50 1],0,27);pmodel13=tfd2step(tfinal,delt,1,g13);pmodel{1,3}=pmodel13(1:N);
p21=poly2tfd(5.39,[50 1],0,18);pmodel21=tfd2step(tfinal,delt,1,g21);pmodel{2,1}=pmodel21(1:N);
p22=poly2tfd(5.72,[60 1],0,14);pmodel22=tfd2step(tfinal,delt,1,g22);pmodel{2,2}=pmodel22(1:N);
p23=poly2tfd(6.09,[40 1],0,15);pmodel23=tfd2step(tfinal,delt,1,g23);pmodel{2,3}=pmodel23(1:N);
p31=poly2tfd(4.38,[33 1],0,20);pmodel31=tfd2step(tfinal,delt,1,g31);pmodel{3,1}=pmodel31(1:N);
p32=poly2tfd(4.42,[44 1],0,22);pmodel32=tfd2step(tfinal,delt,1,g32);pmodel{3,2}=pmodel32(1:N);
p33=poly2tfd(7.20,[19 1],0,0);pmodel33=tfd2step(tfinal,delt,1,g33);pmodel{3,3}=pmodel33(1:N);


%����һ���ṹ�壬�洢ģ�͵ĸ������������ض�����pmodel
steps=100/delt;
controller4.model1=model1;
controller4.P=N;%PΪԤ�ⳤ��
controller4.M=steps; %MΪ����ʱ��
controller4.N=N;%NΪģ�ͳ���
controller4.p=3;  %pΪ�������
controller4.m=3;  %mΪ�������
controller4.pmodel=pmodel;
controller4.alpha=0.01;%�˲�ϵ��
controller4.ek=0;%��ʼ������ʵ������˲�
%��������
controller4.Ti=1;%�����������
controller4.To=3;%�����������
controller4.np=controller4.To/controller4.Ti;%һ�����������Ӧ�������������
controller4.V=controller4.N/controller4.To;%��������/����ʱ��=����Ƶ��
% DMC4:�������ֵ
controller4.Ys=[0.5;-0.5;-0.5];%������趨
[y41,y42,y43,u41,u42,u43]=DMC4(controller4);

controller5.model1=model1;
controller5.P=N;%PΪԤ�ⳤ��
controller5.M=steps; %MΪ����ʱ��
controller5.N=N;%NΪģ�ͳ���
controller5.p=3;  %pΪ�������
controller5.m=3;  %mΪ�������
controller5.pmodel=pmodel;
controller5.alpha=0.01;%�˲�ϵ��
controller5.ek=0;%��ʼ������ʵ������˲�
%��������
controller5.Ti=1;%�����������
controller5.To=10;%�����������
controller5.np=controller5.To/controller5.Ti;%һ�����������Ӧ�������������
controller5.V=controller5.N/controller5.To;%��������/����ʱ��=����Ƶ��
% DMC4:�������ֵ
controller5.Ys=[0.5;-0.5;-0.5];%������趨
[y51,y52,y53,u51,u52,u53]=DMC5(controller5);


controller6.model1=model1;
controller6.P=N;%PΪԤ�ⳤ��
controller6.M=steps; %MΪ����ʱ��
controller6.N=N;%NΪģ�ͳ���
controller6.p=3;  %pΪ�������
controller6.m=3;  %mΪ�������
controller6.pmodel=pmodel;
controller6.alpha=0.01;%�˲�ϵ��
controller6.ek=0;%��ʼ������ʵ������˲�
%��������
controller6.Ti=1;%�����������
controller6.To=20;%�����������
controller6.np=controller6.To/controller6.Ti;%һ�����������Ӧ�������������
controller6.V=controller6.N/controller6.To;%��������/����ʱ��=����Ƶ��
% DMC4:�������ֵ
controller6.Ys=[0.5;-0.5;-0.5];%������趨
[y61,y62,y63,u61,u62,u63]=DMC6(controller6);

figure(1);
plot(y41,'-blue');
hold on;
plot(y51,'-red');
hold on;
plot(y61,'-green');
hold on;
plot(0.5*ones(1,steps),'-.');
hold on;
ylabel('y1 error');
xlabel('t/min');
legend('5T0','10T0','20T0','�ο��켣','location','best');
title('�������ֵ')

figure(2);
plot(y42,'-blue');
hold on;
plot(y52,'-red');
hold on;
plot(y62,'-green');
hold on;
plot(-0.5*ones(1,steps),'-.');
hold on;
ylabel('y2 error');
xlabel('t/min');
legend('5T0','10T0','20T0','�ο��켣','location','best');
title('�������ֵ')

figure(3);
plot(y43,'-blue');
hold on;
plot(y53,'-red');
hold on;
plot(y63,'-green');
hold on;
plot(-0.5*ones(1,steps),'-.');
hold on;
ylabel('y3 error');
xlabel('t/min');
legend('5T0','10T0','20T0','�ο��켣','location','best');
title('�������ֵ')
    
    
