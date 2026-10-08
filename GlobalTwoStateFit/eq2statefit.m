clear
clc

load farUVCD.dat; % WT, S26D, S41D, T63D, 26/41, 26/63, 41/63, TM.. signal in MRE units at 222 nm
amp=farUVCD;

T=amp(:,1); % in K
datx=amp(:,2:end);
[a np]=size(datx);

dataini=datx./1000;

for i=1:np
    dataini2(a*(i-1)+1:a*i,1)=dataini(:,i);
    T2(a*(i-1)+1:a*i,1)=T;
end

v=[-18 0.001 -5 -0.01 1 320*ones(1,np) 120*ones(1,np)]'; % initial parameter guess
    
lw=[-inf  0  -inf -inf 0   0*ones(1,np)    0*ones(1,np)]'; % lower bound
up=[0    inf  inf   0  10 340*ones(1,np)   400*ones(1,np)]'; % upper bound


options=optimset('lsqcurvefit');
options.Display='iter';
options.MaxFunEvals=1e7;
options.MaxIter=1e7;
options.TolX=1e-6; % the default values for both are 1e-6
options.TolFun=1e-6;   

[p,resnorm,residual,exitflag,output,lambda,jacobian]=lsqcurvefit('eq2stateIndFit',v,T2,dataini2,lw,up,options); 
[fit,sig,pf,pu,dG,Sf,Su]=eq2stateIndFit(p,T2);
ci=nlparci(p,residual,jacobian); % to get the 95 % conf int for parameters


figure
plot(T,dataini,'o',T,sig,'-',T,Sf,'g',T,Su,'m')
xlabel('Temperature (K)')
ylabel('MRE @ 222 nm (x 10-3)')

figure
plot(T,pf,'-')
xlabel('Temperature (K)')
ylabel('Folded state probility')

parfin(:,i)=p;

Tmvfin=p(6:13);
dHmvfin=p(14:21);
figure
plot(Tmvfin,dHmvfin,'bo')
xlabel('T_m (K)')
ylabel('deltaH_m (kJ mol-1 K-1)')

for i=1:length(T)
    pfstd(i,1)=std(pf(i,:));
end
figure
plot(T,pfstd,'bo')
xlabel('Temperature (K)')
ylabel('Standard deviation of folded state probability')

figure
plot(dG(5,:),'bo')
xlabel('Mutant Index')
ylabel('deltaG (kJ mol-1)')

[ci(:,1) p ci(:,2) abs(ci(:,2)-p)]