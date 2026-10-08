
function [funct,probf,probu,Cpf,Cpu,dcpave,cpex,d2have,dh2ave]=TwoStateDSCfit(v,T,dataini)

% this fits the DSC data to a 2-state model .. 

dHo=v(1); % vant hoff enthalpy
Tm=v(2); % melting temperature
af=v(3);
bf=v(4);
au=v(5);
bu=v(6);

R=0.008314;

To=298;
Cpf=af+bf.*(T-To); % heat capacity of the folded state
Cpu=au+bu.*(T-To); % heat capacity of the unfolded state

dCp=Cpu-Cpf;

dH=dHo+dCp.*(T-Tm);
dS=(dHo/Tm)+dCp.*log(T./Tm);
dG=dH-T.*dS;
K=exp(-dG./(R*T));
probf=1./(1+K);
probu=K./(1+K); % taking the unfolded state as the reference.. 

dHca=dH;
dcpave=Cpu.*probu+Cpf.*probf; % chemical baseline
d2have=(dHca.^2).*probu;
dh2ave=dHca.*probu;
cpex=(d2have-dh2ave.^2)./(R.*(T.^2)); % transition heat capacity
dsc=dcpave+cpex; 

funct=dsc;