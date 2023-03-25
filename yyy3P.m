% <<Comprehensive holographic internal beam modulation with spherical aberration compensation>>
% the followed codes are used in article.
% target distribution is three foci (equal energy).
%% parameter definetion
clear;
close all;
warning off;
global lambda k pm pn NA f R L0 dx Lx Ly n1 n2 d pz
lambda = 1064e-6;
k = 2*pi/lambda;
pm = 860;
pn = 860;
NA = 0.671875;
f = 4;
R = NA * f;
L0 = 2 * R;
dx = L0 ./ pm;
Lx = pm * dx;
Ly = pn * dx;
n1 = 1;
n2 = 1.44;
d = 1.2872;
pz = 3;
useCUDA = 1;
useLBFGS = 0;
useWholePolarization = 0;

if useWholePolarization == 1
    [Jax,Jay,Jaz,Q3d,ap] = Fx_coefficient_environment_xyz(0.7128,70e-3);
else
    [Jax,Q3d,ap] = Fx_coefficient_environment(0.7128,70e-3);
end

Ein = Fx_gaussianbeam(pm,pn,3.5/2,dx,1) .* ap;
energySF = Fx_energySingleFocus(Ein,ap,Jax);
%% target field definetion
target0 = zeros(pm,pn);
target0(pm/2+1,pm/2+1) = 1;
T3d = zeros(pm,pn,pz);
for ii = 1:3
    T3d(:,:,ii) = circshift(target0,[0,3*(ii-2)]);
end
T3d = T3d * energySF / sum(T3d,"all") * 1.05;
weight3d = 80 * T3d ./ max(T3d,[],"all") + 1;  %80good
%% optimization preparetion
if useLBFGS == 0
    W = ones(pm,pn);
    veloc = zeros(pm,pn);
    if useCUDA == 1
        W = gpuArray(W);
        Ein = gpuArray(Ein);
        Jax = gpuArray(Jax);
        Q3d = gpuArray(Q3d);
        T3d = gpuArray(T3d);
        weight3d = gpuArray(weight3d);
        veloc = gpuArray(veloc);
    end
end

if useLBFGS == 1
    phase0 = ones(pm,pn);
    if useCUDA == 1
        Ein = gpuArray(Ein);
        Jax = gpuArray(Jax);
        Q3d = gpuArray(Q3d);
        weight3d = gpuArray(weight3d);
        T3d = gpuArray(T3d);
    end
end

%% L-BFGS
if useLBFGS == 1
    tic
    if useWholePolarization == 1
        cost_FxandGr = @(W)Fx_costFxandGr_LBFGS_xyz(W,Ein,Jax,Jay,Jaz,Q3d,weight3d,T3d,useCUDA);
    else
        cost_FxandGr = @(W)Fx_costFxandGr_LBFGS(W,Ein,Jax,Q3d,weight3d,T3d,useCUDA);
    end

    matlab_options = optimoptions('fmincon','GradObj','on', 'display', 'iter', ...
        'algorithm','interior-point','HessianApproximation','lbfgs', 'MaxFunEvals', 300, 'MaxIterations', 300,...
        'TolX', 1e-20, 'TolFun', 1e-12);
    lb = -inf(pm,pn);
    ub = inf(pm,pn);

    W = fmincon(cost_FxandGr,phase0,[],[],[],[],lb,ub,[],matlab_options);
    toc
end
%% gradient descent (with momentum)
if useLBFGS == 0
    LR = 2e-22;    %240e-22
    momentum = 0.95;
    tic
    for ii = 1:300
        if useWholePolarization == 1
            [finalG3d,Wbar] = Fx_costFxandGr_GDM_xyz(Ein,W,Jax,Jay,Jaz,Q3d,T3d,weight3d);
        else
            [finalG3d,Wbar] = Fx_costFxandGr_GDM(Ein,W,Jax,Q3d,T3d,weight3d);
        end
        costValue(ii) = finalG3d;

        fang = - Wbar + momentum * veloc;
        W = W + LR * fang;
        if ii > 2
            veloc = fang;
        end
        if mod(ii,50) == 0
            ii
            costValue = gather(costValue);
        end
    end
    toc
    
end
%% observation 
Wideal = W;
E = Ein .* exp(1i * k * Wideal);
if useWholePolarization == 1
    F3d = Fx_ObserveFocusRegion_xyz(E,1,Jax,Jay,Jaz,Q3d);
else
    F3d = Fx_ObserveFocusRegion(E,1,Jax,Q3d);
end
%% Generating CGH for SLM (Hamamatsu 15223)
xx = linspace(-0.5*12.5e-3*1025,0.5*12.5e-3*1025,1025);
yy = linspace(-0.5*12.5e-3*1273,0.5*12.5e-3*1273,1273);
[X,Y] = meshgrid(xx,yy);
[theta,r] = cart2pol(X,Y);
grating = exp(1i .* 75 * (X+Y));
grating(r<6.40625/1024*860) = 0;
figure;
imagesc(angle(grating))

SLM0 = zeros(1025,1273);
SLM0(1024/2-(pm/2):1024/2+(pm/2-1),1272/2-(pm/2):1272/2+(pm/2-1)) = exp(1i * k * Wideal);
SLM0 = SLM0.';
SLM0(r>6.40625/1024*860) = 0;
figure;
mesh(angle(SLM0))
SLM = SLM0.' + grating.';
SLM = circshift(SLM,0,1);
SLM = circshift(SLM,-30,2);
figure;
imagesc(angle(SLM))
pic1 = (angle(SLM)+pi) / 2 / pi * 256;
pic1 = uint8(pic1);
imwrite(pic1,'3P 2000.bmp')
