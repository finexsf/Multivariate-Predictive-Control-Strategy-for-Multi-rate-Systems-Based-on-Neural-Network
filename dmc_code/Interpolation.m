function a=Interpolation(b0,np,V)
seta=zeros(3*V,1);%seta:二次多项式系数矩阵
%b0:V个输出采样点

%构造二次样条插值系数矩阵Z
z11=zeros(1,3*V);
z11(1)=1;

for i=1:V
    zz121{i}=[1,i*np,(i*np)^2];
end
z21=blkdiag(zz121{1:V});

zz131=zeros(V-1,3);
zz132=z21(1:(V-1),1:(3*V-3));
z31=cat(2,zz131,zz132);

Z1=cat(1,z11,z21,z31);

for q=1:V
    zz211{q}=[0,1,2*q*np];
end
zz212=blkdiag(zz211{1:V});
zz213=zeros(V,3);
zz214=cat(2,zz213,zz212);
zz214(V,:)=0;
zz215=-zz214(:,1:(3*V));

Z2=zz212+zz215;

Z=cat(1,Z1,Z2);

%构造输出向量b
b1=b0(1:(V-1),1);
b2=zeros(V,1);
b=cat(1,0,b0,b1,b2);

%求解二次多项式参数
seta=inv(Z)*b;

for g=1:V
    if b0(g)~=0
        break
    end
end
%求解基准周期下的预测模型输出
for j=g+1:V
    for i=1:np-1
      a1(j,i)=seta(j*3,1)*(np*j-(np-i))^2+seta(j*3-1,1)*(np*j-(np-i))+seta(j*3-2,1);
    end
end
a(1)=b0(1);
for k=1:V
    for g=1:np*V  
        if g==np*k &&g>1
            a(g)=b0(k);
        for i=1:np-1
            a(g-i)=a1(k,np-i);
        end
        end
    end
end

a=a';
end