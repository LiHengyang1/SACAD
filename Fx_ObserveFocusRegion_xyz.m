function [F3d] = Fx_ObserveFocusRegion_xyz(Ein,ratio,Jax,Jay,Jaz,Q3d)
global   pz pm

B3dx = repmat(Ein .* Jax,[1,1,pz]) .* Q3d;
B3dy = repmat(Ein .* Jay,[1,1,pz]) .* Q3d;
B3dz = repmat(Ein .* Jaz,[1,1,pz]) .* Q3d;

for jj = 1:pz
    D3dx(:,:,jj) = Fx_CZT(B3dx(:,:,jj),ratio,pm);
end
for jj = 1:pz
    D3dy(:,:,jj) = Fx_CZT(B3dy(:,:,jj),ratio,pm);
end
for jj = 1:pz
    D3dz(:,:,jj) = Fx_CZT(B3dz(:,:,jj),ratio,pm);
end

F3dx = abs(D3dx).^2;
F3dy = abs(D3dy).^2;
F3dz = abs(D3dz).^2;
F3d = F3dx + F3dy + F3dz;
F3d = gather(F3d);
Wideal = gather(angle(Ein));
figure;
imagesc(angle(exp(1i * Wideal)))
for ii = 1:pz
    figure;
    mesh(F3d(:,:,ii));
end
end