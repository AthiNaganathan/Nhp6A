function [funct,sig,pf,pu,dG,Sf,Su]=eq2stateIndFit(v,T2)

np=8;
T=[5:5:95]'+273.15;

RT=0.008314*T;
Tref=298.15;

Sf0=v(1);
Sf1=v(2);
Su0=v(3);
Su1=v(4);
Sf=Sf0+Sf1*(T-Tref); % Shared folded baseline
Su=Su0+Su1*(T-Tref); % Shared unfolded baseline

dCp=v(5); % Heat capacity change

Tmv=v(6:13); % Melting temperature
dHmv=v(14:21); % Enthalpy of unfolding at the melting temperature

for i=1:np

    dG(:,i)=dHmv(i,1)+dCp*(T-Tmv(i,1))-T.*((dHmv(i,1)./Tmv(i,1))+dCp*log(T/Tmv(i,1))); % Gibbs-Helmholtz relation
    Ku(:,i)=exp(-dG(:,i)./RT); % Equilibrium constant
    pf(:,i)=1./(1+Ku(:,i)); % Probability of the folded state
    pu(:,i)=Ku(:,i)./(1+Ku(:,i)); % Probability of the unfolded state
    sig(:,i)=pf(:,i).*Sf+pu(:,i).*Su; % Predicted signal
    
end

calc=reshape(sig,19*np,1);

funct=calc;
