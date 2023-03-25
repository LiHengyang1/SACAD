function [finalG3d,Wbar] = Fx_costFxandGr_GDM_xyz(Ein,W,Jax,Jay,Jaz,Q3d,T3d,weight3d)
global k pz

E = Ein .* exp(1i * k * W);
D3dx = fftshift(fftshift(fft2(repmat(E .* Jax,[1,1,pz]) .* Q3d),1),2);
D3dy = fftshift(fftshift(fft2(repmat(E .* Jay,[1,1,pz]) .* Q3d),1),2);
D3dz = fftshift(fftshift(fft2(repmat(E .* Jaz,[1,1,pz]) .* Q3d),1),2);
F3d = abs(D3dx).^2 + abs(D3dy).^2 + abs(D3dz).^2;

finalG3d = sum((F3d - T3d).^2 .* weight3d,'all');

F3dbar = 2 * weight3d .* (F3d - T3d);
C3dbarx = ifft2(ifftshift(ifftshift(2 * D3dx .* F3dbar,1),2)) .* conj(Q3d);
C3dbary = ifft2(ifftshift(ifftshift(2 * D3dy .* F3dbar,1),2)) .* conj(Q3d);
C3dbarz = ifft2(ifftshift(ifftshift(2 * D3dz .* F3dbar,1),2)) .* conj(Q3d);
Wbar = k * Ein .* imag((sum(C3dbarx,3) .* conj(Jax) + sum(C3dbary,3) .* conj(Jay) + sum(C3dbarz,3) .* conj(Jaz)) .* conj(E));

end