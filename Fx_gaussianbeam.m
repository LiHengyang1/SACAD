function field = Fx_gaussianbeam(N,M,R_gaussian,dx,E_summit)
%  生成高斯光束，相位为0
%  input:  N，M一行（一列）分辨率， R_gaussian高斯半径  E_summit最高点（中心）场强
%  output：field高斯广场
    
    if nargin == 5;
    end
    if nargin == 4;
        E_summit = 1; 
    end
    gridbase1 = ([0 : N - 1] - (N - 1) / 2).';
    gridbase2 = ([0 : M - 1] - (M - 1) / 2).';
    [U,V] = meshgrid(gridbase1,gridbase2);
    field = E_summit * exp(-(U.^2 + V.^2) / (R_gaussian / dx).^2);
end