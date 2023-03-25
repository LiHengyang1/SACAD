function output = Fx_CZT(input,ratio,m)

w = 1.0 * exp(-1j * 2 * pi * ratio / m );
a = - exp(1j * 2 * pi * (0.5-0.5*ratio));
output1 = czt(input,m,w,a);
output = czt(output1.',m,w,a);
output = output.';
end
