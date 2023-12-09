function x = recover_integer(w,z_opt,y_opt,H,P,num)
x = zeros(num,1);
H = H.*H;
for i = 1:num
    sum = 0;
    for j = 1:num
        sum = sum + y_opt(j)^2*H(j,i)*P*1;
    end
    Q = 2*y_opt(i)*sqrt(w(i)*(1+z_opt(i))*H(i,i)*P*1)-sum;
    if Q>0
        x(i) = 1;
    end    
end
end