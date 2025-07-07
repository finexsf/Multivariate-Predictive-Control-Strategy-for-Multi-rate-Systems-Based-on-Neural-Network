function [K,T]=Foiemodel(b1,b2,np)
%求解阶跃响应对应的系数
T=np./log(1-b1./b2);
K=b2;
end
