function [X,Y,Distance,Distance_quan] = generate(num,data_size)
iteration = zeros(data_size,1);
X=zeros(num^2,data_size);
Y=zeros(num,data_size);
T_layout=zeros(num,2*data_size);
R_layout=zeros(num,2*data_size);
pair_d=zeros(num,data_size);
Distance=zeros(num^2,data_size);
Distance_quan=zeros(num^2,data_size);

carrier_f = 2.4e+9;
antenna_h = 1.5;
P_dBm = 40; %transmit power level
P = dB_trans(P_dBm-30);
x_int = ones(num,1);
noise_density = -169;%dBm/Hz
bandwidth = 5e+6;
p_noise_dBm = noise_density + 10*log10(bandwidth);
p_noise = dB_trans(p_noise_dBm-30);
length = 500;
min_d = 2;
max_d = 65;
for loop = 1:data_size
    [Tx,Ty,Rx,Ry,pair_dis_original] = create_random_location(length,num,min_d,max_d);
    [channel_h,d_original] = channel_fading(carrier_f,antenna_h,Tx,Ty,Rx,Ry,num);
    %quantization
    pair_distance = quantization8(pair_dis_original,min_d,max_d);
    d =  quantization8(d_original,0,length);
    for i = 1:num
        d(i,i)=pair_distance(i);
    end
    w = ones(num,1);
    [x_opt,z_opt,y_opt,obj] = FPlinQ_sum_rate_change(num,w,x_int,channel_h,P,p_noise);
    x = recover_integer(w,z_opt,y_opt,channel_h,P,num);
    channel_h_column = reshape(channel_h,num^2,1);
    d_column = reshape(d,num^2,1);
    d_original_column = reshape(d_original,num^2,1);
    X(:,loop) = channel_h_column;
    Y(:,loop) = x;
    T_layout(:,2*loop-1) = Tx;
    T_layout(:,2*loop) = Ty;
    R_layout(:,2*loop-1) = Rx;
    R_layout(:,2*loop) = Ry; 
    pair_d(:,loop) = pair_distance; 
    Distance_quan(:,loop) = d_column;
    Distance(:,loop) = d_original_column;
end
end