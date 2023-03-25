function [Jax,Jay,Jaz,Q3d,ap] = Fx_coefficient_environment_xyz(z0,interval)

global lambda k pm pn NA f R L0 dx Lx n1 n2 d pz
k = 2*pi/lambda;
R = NA * f;
L0 = 2 * R;
dx = L0 ./ pm;
dy = L0 ./ pn;
Lx = pm * dx;
Ly = pn * dx;


xx = linspace(-0.5*Lx,0.5*Lx,pm);
yy = linspace(-0.5*Ly,0.5*Ly,pn);
[X,Y] = meshgrid(xx,yy);
[phi,r] = cart2pol(X,Y);

ap = zeros(pm,pn);
ap(r<R) = 1;
thata1 = asin(r ./ f);
theta2 = asin(n1 / n2 * sin(thata1));
taoP = 2 * sin(theta2) .* cos(thata1) ./ (sin(thata1 + theta2) .* cos(thata1 - theta2)) .* ap;
taoS = 2 * sin(theta2) .* cos(thata1) ./ sin(thata1 + theta2) .* ap;

a11 = taoP .* cos(theta2) .* (cos(phi)).^2 + taoS .* (sin(phi)).^2;
a12 = (taoP .* cos(theta2) - taoS) .* cos(phi) .* sin(phi);
a13 = -taoP .* cos(phi) .* sin(theta2);

cx = f * sqrt(cos(thata1)) .* a11;
cy = f * sqrt(cos(thata1)) .* a12;
cz = f * sqrt(cos(thata1)) .* a13;

phase_surf = exp(1i * k * d * (n2 * cos(theta2) - n1 * cos(thata1)));

Jax = cx .* phase_surf ./ cos(thata1);
Jay = cy .* phase_surf ./ cos(thata1);
Jaz = cz .* phase_surf ./ cos(thata1);

Jax(isnan(Jax)) = 0;
Jay(isnan(Jay)) = 0;
Jaz(isnan(Jaz)) = 0;

Q3d = zeros(pm,pn,pz);
for ii = 1:pz
    z = z0 + interval  * (ii - (floor(pz/2)+1));
    phase_prop = exp(1i * k * n2 * cos(theta2) * z);
    Q3d(:,:,ii) = phase_prop;
end

end