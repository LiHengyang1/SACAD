function [F3d] = Fx_ObserveFocusRegion(Ein,ratio,Jax,Q3d)
global  k pz pm

C = Ein .* Jax;
C3d = repmat(C,[1,1,pz]);
B3d = C3d .* Q3d;
for jj = 1:pz
    D3d(:,:,jj) = Fx_CZT(B3d(:,:,jj),ratio,pm);
end
F3d = abs(D3d).^2;
F3d = gather(F3d);
Wideal = gather(angle(Ein));
figure;
imagesc(angle(exp(1i * Wideal)))
for ii = 1:pz
    figure;
    mesh(F3d(:,:,ii));
end
end