% Simulating the different Tm-DHm scenarios

clear all
clc

nres=67; % no. of folded residues in Nhp6A
R=0.008314;
T=[278:1:368]'; % temperature range
Tmx=[333 333 313 313 313]'; % melting temperature
DHmx=[2.9 1.45 1.45 2.9 2.9]'*nres; % enthalpy of unfolding at the midpoint
DCp=(40./1000)*nres; % heat capacity change

for i=1:length(Tmx)

    Tm=Tmx(i,1);
    DHm=DHmx(i,1);
    DTm=4;
    Tmx2=[Tmx(i,1) Tmx(i,1)-DTm*1 Tmx(i,1)-DTm*2 Tmx(i,1)-DTm*3 Tmx(i,1)-DTm*4]'; % Tm of the mutants
    DHmx2=ones(length(Tmx2),1)*DHm;

    for j=1:length(Tmx2)

        if i==5
            DHmx2=[DHmx(i,1) DHmx(i,1)*0.9 DHmx(i,1)*0.8 DHmx(i,1)*0.7 DHmx(i,1)*0.6]'; % DHm of the mutants
        end
            
        DG(i,j,:)=DHmx2(j,1)+DCp*(T-Tmx2(j))-T.*((DHmx2(j,1)./Tmx2(j)+DCp.*log(T./Tmx2(j)))); % stability as a function of temperature

        Keq(i,j,:)=exp(-squeeze(DG(i,j,:))./(R.*T)); % equilibrium constant

        pf(i,j,:)=1./(1+Keq(i,j,:)); % folded state probability

    end
end

subplot(2,3,1)
plot(T,squeeze(pf(1,:,:)),'-')
yyaxis left 
hold
plot([303 303],[0 1],'k--')
axis([280 350 0 1])
xlabel('Temperature (K)'); ylabel('Probability of folded state')
yyaxis right
plot(T,std(squeeze(pf(1,:,:))),'k')
axis([280 350 0 0.4])
title('High DHm, High Tm')

subplot(2,3,2)
plot(T,squeeze(pf(2,:,:)),'-')
yyaxis left 
hold
plot([303 303],[0 1],'k--')
axis([280 350 0 1])
xlabel('Temperature (K)'); ylabel('Probability of folded state')
yyaxis right
plot(T,std(squeeze(pf(2,:,:))),'k')
axis([280 350 0 0.4])
title('Low DHm, High Tm')

subplot(2,3,3)
plot(T,squeeze(pf(3,:,:)),'-')
yyaxis left 
hold
plot([303 303],[0 1],'k--')
axis([280 350 0 1])
xlabel('Temperature (K)'); ylabel('Probability of folded state')
yyaxis right
plot(T,std(squeeze(pf(3,:,:))),'k')
axis([280 350 0 0.4])
title('Low DHm, Low Tm')

subplot(2,3,4)
plot(T,squeeze(pf(4,:,:)),'-')
yyaxis left 
hold
plot([303 303],[0 1],'k--')
axis([280 350 0 1])
xlabel('Temperature (K)'); ylabel('Probability of folded state')
yyaxis right
plot(T,std(squeeze(pf(4,:,:))),'k')
axis([280 350 0 0.4])
title('High DHm, Low Tm')

subplot(2,3,5)
plot(T,squeeze(pf(5,:,:)),'-')
yyaxis left 
hold
plot([303 303],[0 1],'k--')
axis([280 350 0 1])
xlabel('Temperature (K)'); ylabel('Probability of folded state')
yyaxis right
plot(T,std(squeeze(pf(5,:,:))),'k')
axis([280 350 0 0.4])
title('Varying high DHm, Low Tm')
