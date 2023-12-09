function [Tx,Ty,Rx,Ry,pair_distance]= create_random_location(length,num,min_d,max_d)
Tx = unifrnd(0,length,num,1);
Ty = unifrnd(0,length,num,1);
Rx = zeros(num,1);
Ry = zeros(num,1);
pair_distance = zeros(num,1);
for i=1:num
    flag = true;
    while(flag)
      theta = 2*pi*rand(1,1);
      r = (max_d-min_d)*rand(1,1)+min_d;
      pair_distance(i) = r;
      x = r.*cos(theta);
      y = r.*sin(theta);
      Rx(i) = Tx(i)+x;
      Ry(i) = Ty(i)+y;
      if Rx(i)>=0&&Rx(i)<=length&&Ry(i)>=0&&Ry(i)<=length
          flag = false;
      end
    end
end