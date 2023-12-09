function [x_opt,z_opt,y_opt,obj] = FPlinQ_sum_rate_change(num,w,x_int,H,P,p_noise)
iter = 0;
x = x_int;
x_relax = zeros(num,1);
z = zeros(num,1);
y = zeros(num,1);
H = H.*H;

sum_log = 0;
sum_wz = 0;
sum_ratio = 0;
sum_total = 0;
for i = 1:num
    sum = 0;
    for j = 1:num
        sum = sum + H(j,i)*P*x(j);
    end
    sum_ij = sum - H(i,i)*P*x(i); 
    z(i) = H(i,i)*P*x(i)/(sum_ij+p_noise);
    y(i) = sqrt(w(i)*(1+z(i))*H(i,i)*P*x(i))/(sum+p_noise);

    sum_log = sum_log+w(i)*log2(1+z(i));
    sum_wz = sum_wz+w(i)*z(i);
    sum_ratio = sum_ratio+w(i)*(1+z(i))*H(i,i)*P*x(i)/(sum+p_noise);
    sum_total = sum_total+sum_log-sum_wz+sum_ratio;
    
end
object = [sum_total];


while(1)  
    iter = iter+1;
    sum_total_old = sum_total;
 
    for i = 1:num
        sum = 0;
        for j = 1:num
            sum = sum + y(j)^2*H(i,j)*P;
        end
        x_relax(i) = (y(i)*sqrt(w(i)*(1+z(i))*H(i,i)*P)/sum)^2;
        x(i) = min([1,x_relax(i)]);
    end
    sum_log = 0;
    sum_wz = 0;
    sum_ratio = 0;
    sum_total = 0;
    for i = 1:num
        sum = 0;
        for j = 1:num
            sum = sum + H(j,i)*P*x(j);
        end
        sum_ij = sum - H(i,i)*P*x(i);

        z(i) = H(i,i)*P*x(i)/(sum_ij+p_noise);

        y(i) = sqrt(w(i)*(1+z(i))*H(i,i)*P*x(i))/(sum+p_noise);

        sum_log = sum_log+w(i)*log2(1+z(i));
        sum_wz = sum_wz+w(i)*z(i);
        sum_ratio = sum_ratio+w(i)*(1+z(i))*H(i,i)*P*x(i)/(sum+p_noise);
        sum_total = sum_total+sum_log-sum_wz+sum_ratio;
    end
    object = [object sum_total];


    if abs((sum_total-sum_total_old)/sum_total_old)<=5e-4 || iter>100
        break;
    end
        
end

x_opt = x;
y_opt = y;
z_opt = z;
obj = object;
end
