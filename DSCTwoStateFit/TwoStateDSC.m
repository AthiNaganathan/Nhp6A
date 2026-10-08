clear all
clc

load Cpabs_Nhp6a.dat; % [Temperature(K) Cp(kJ mol-1 K-1)] .. load the DSC data
datai=Cpabs_Nhp6a; 

dataini=datai(:,2); 

[row column]=size(dataini);
T=datai(:,1);

v=[180 333 20 0.001 25 -0.0001]'; % initial parameter guesses
lw=[]; % lower bound
up=[]; % upper bound

w=ones(row,1); % weights (not considered)

options=optimset('lsqcurvefit');
options.Display='iter';
options.TolX=1e-9;
options.TolFun=1e-9;
options.MaxFunEvals=10e4;

[p,resnorm,residual,exitflag,output,lambda,jacobian]=lsqcurvefit('TwoStateDSCfit',v,T,dataini,lw,up,options,dataini);
v=p;
[fit,probf,probu,Cpf,Cpu,dcpave,cpex,d2have,dh2ave]=TwoStateDSCfit(v,T,dataini);

sls=sum((dataini-fit).^2)

Tdsc=T;

plot(Tdsc,dataini,'bo',Tdsc,fit,'r',Tdsc,Cpf,'g',Tdsc,Cpu,'m',Tdsc,dcpave,'k');
xlabel('Temperature (K)')
ylabel('C_p (kJ mol-1 K-1)')

R=0.008314;
Tb=[0:2:100]'+273.15;
dG=p(1)-Tb.*(p(1)./p(2));
K=exp(-dG./(R*Tb));
pf=1./(1+K); % folded state probability

ci=nlparci(p,residual,jacobian); % to get the 95 % conf int for parameters

[ci(:,1) p ci(:,2) abs(ci(:,2)-p)] 

sls=sqrt(sum((dataini-fit).^2))

