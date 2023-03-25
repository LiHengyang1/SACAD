function energySF = Fx_energySingleFocus(Ein,ap,Jax)
global  k pm pn NA f Lx Ly n1 n2 d
xx = linspace(-0.5*Lx,0.5*Lx,pm);
yy = linspace(-0.5*Lx,0.5*Ly,pn);
[X,Y] = meshgrid(xx,yy);
[phi,r] = cart2pol(X,Y);
theta1 = asin(r ./ f);

dnorm = d;
rou = n1 * sin(theta1) / NA .* ap;
Dn2 = k * dnorm * (sqrt(n2^2 - (NA*rou).^2)) .* ap;
Dn2P = Dn2 - mean(Dn2(Dn2>0.01));
phiSA = -k * dnorm * (sqrt(n2^2 - (NA*rou).^2) - sqrt(n1^2 - (NA*rou).^2)) .* ap;
phiSAP = phiSA - mean(phiSA(phiSA<-0.01));
s = 1 / (1 + sum(phiSAP .* Dn2P .* ap,"all") / sum(Dn2P .* Dn2P .* ap,"all"));
dact = dnorm / s;
SApurephase = k * dnorm / s * (s * sqrt(n1^2 - (NA*rou).^2) - sqrt(n2^2 - (NA*rou).^2)) .* ap;

thata1 = asin(r ./ f);
theta2 = asin(n1 / n2 * sin(thata1));
z = dact - dnorm;
phase_prop = exp(1i * k * n2 * cos(theta2) * z);

E = Ein .* exp(1i * SApurephase);
D3d = fftshift(fftshift(fft2(E .* Jax .* phase_prop),1),2);
F3d = abs(D3d).^2;
energySF = max(F3d,[],"all");
% figure;
% mesh(F3d)
end


% Reference:
% DOI:10.1364/OE.22.017644  <<Exploring the depth range for
% three-dimensional laser machining with aberration correction>>