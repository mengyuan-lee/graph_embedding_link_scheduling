function quan = quantization4(input,min,max)

quan = input;

quan(input<min+(max-min)/4)=0;
quan(min+(max-min)/4<input&input<min+(max-min)/4*2)=1;
quan(min+(max-min)/4*2<input&input<min+(max-min)/4*3)=2;
quan(min+(max-min)/4*3<input)=3;


end