function [channel_h,d] = channel_fading(carrier_f,antenna_h,Tx,Ty,Rx,Ry,num)
c=3e+8; 
lambda = c./carrier_f; 
Rbp = 4*antenna_h^2/lambda; 
Lbp_dB = 20*log10(lambda^2/(8*pi*antenna_h^2));
Lbp_dB = abs(Lbp_dB);
d = zeros(num);
for i=1:num
    for j=1:num
        d(i,j) = sqrt((Tx(i)-Rx(j))^2+(Ty(i)-Ry(j))^2);
    end
end
Loss = zeros(num);
channel_h = zeros(num);
for i=1:num
    for j=1:num        
        if d(i,j)<=Rbp
            Loss(i,j) = Lbp_dB+6+20*log10(d(i,j)/Rbp);
            channel_h(i,j) = dB_trans(- Loss(i,j));
        else
            Loss(i,j) = Lbp_dB+6+40*log10(d(i,j)/Rbp);
            channel_h(i,j) = dB_trans(- Loss(i,j));
        end
    end
end

