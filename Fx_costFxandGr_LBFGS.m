function [finalG3d,Wbar] = Fx_costFxandGr_LBFGS(W,Ein,Jax,Q3d,weight3d,T3d,useCUDA)
global  k pz
if useCUDA == 1
    W = gpuArray(W);
end

E = Ein .* exp(1i * k * W);
D3d = fftshift(fftshift(fft2(repmat(E .* Jax,[1,1,pz]) .* Q3d),1),2);
F3d = abs(D3d).^2;
finalG3d = sum((F3d - T3d).^2 .* weight3d,'all');

C3dbar = ifft2(ifftshift(ifftshift(4 * D3d .* weight3d .* (F3d - T3d),1),2)) .* conj(Q3d);
Wbar = k * imag(sum(C3dbar,3) .* conj(Jax) .* conj(E));
if useCUDA == 1
    finalG3d = gather(finalG3d);
    Wbar = gather(Wbar);
end
end