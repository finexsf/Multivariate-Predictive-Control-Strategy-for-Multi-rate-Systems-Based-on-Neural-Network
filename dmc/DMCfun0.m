%----------------单变量动态矩阵（DMC）算法
function controller=DMCfun0(controller)

%----无约束DMC算法
    N = controller.N;     
    P = controller.P; %P为预测时域    
    M = controller.M; %M为控制时域     
    p = controller.p; %p为输出个数     
    m = controller.m; %m为输入个数         
    y = controller.y; %当前输出     
    lv =controller.lv;%滤波系数
    Ys =controller.Ys;%输出点设定
    k  =controller.k;%k=1
    np =controller.np;%一个输出采样对应的输入采样数量
    V =controller.V;%采样点数/采样时间=采样频率
%----生成动态矩阵A
model = controller.model1;

if isempty(controller.A)
    %-----%动态矩阵
    for i=1:p      
        for j=1:m
            for i1=1:M
                for i2=1:P
                    if i2<i1
                        A{i,j}(i2,i1)=0;
                    else
                        A{i,j}(i2,i1)=model{i,j}(i2-i1+1);
                    end
                end
            end
        end
        C{i}=cat(2,A{i,1:m});
    end
    controller.A=cat(1,C{1:p});
%----计算一步预测矩阵A0
    for i=1:p
        aa{i}=cat(2,model{i,1:m});%阶跃响应系数模型
    end
    A0=cat(1,aa{1:p});%一步预测矩阵
    controller.A0 = A0;
%----计算移位矩阵S0
 S1=zeros(N);
 for i=1:N
    for j=1:N
        if j==i+1
            S1(i,j)=1;
        end
    end
 end
 S1(N,N)=1;
 for i=1:p
     S{i}=S1;
 end
 controller.S0=blkdiag(S{1:p});
%----初始化模块
    controller.ym=zeros(p*N,1);
%----选择校正系数
    hst=ones(N,1);
 for i=1:p
     hh{i}=hst;
 end
 controller.H=blkdiag(hh{1:p});
%----即时增量提取矩阵
 L=zeros(m,m*M);
 for i=1:m
     L(i,(i-1)*M+1)=1;
 end
 controller.L=L;
%----第一个预测值提取矩阵
 SS=zeros(p,p*N);
 for i=1:m
     SS(i,(i-1)*N+1)=1;
 end
 controller.SS=SS;
    
    A=controller.A;
    S0=controller.S0;
    H=controller.H;
    L=controller.L;
    SS=controller.SS;
    
%----生成加权系数矩阵
 erwecoe_y=1;
 erwecoe_u=1;
 q=erwecoe_y*eye(P);
 r=erwecoe_u*eye(M);
 for i=1:p
     qq{i}=q;
 end
 Q=blkdiag(qq{1:p});
 for i=1:m
     rr{i}=r;
 end
 R=blkdiag(rr{1:m});
 %----计算控制量增益D
 controller.D=L*inv(A'*Q*A+R)*A'*Q;
end
    
%----在线计算控制增量du
    
%----求取参考轨迹
for i=1:p
     w{i}=Ys(i)*ones(P,1);
 end
 for i=1:p
     Ws{i}=[filter([0 (1-controller.lv(i))],[1 -controller.lv(i)],w{i},y(i))];
 end
 controller.W=cat(1,Ws{1:p});
 %反馈矫正部分
 yn=controller.ym+controller.A0*controller.du;
 alpha=controller.alpha;
    for kp=1:V
        if k==np*kp
            e=alpha*(y-controller.SS*yn)+(1-alpha)*controller.ek;
        else
            e=alpha*[0;0;0]+(1-alpha)*controller.ek;
        end
    end
    controller.ek=e;
    ycor=yn+controller.H*e;
    controller.ym=controller.S0*ycor;
for i=1:p
     pp{i}=controller.ym(N*(i-1)+1:N*(i-1)+P);
 end
 yp0=cat(1,pp{1:p});
    controller.du=controller.D*(controller.W-yp0);