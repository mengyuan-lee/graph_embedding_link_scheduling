function quan = quantization8(input,min,max)

quan = input;
if min==max
    quan = zeros(size(input));
else
    quan(input<min+(max-min)/8)=0;
    quan(min+(max-min)/8<input&input<min+(max-min)/8*2)=1;
    quan(min+(max-min)/8*2<input&input<min+(max-min)/8*3)=2;
    quan(min+(max-min)/8*3<input&input<min+(max-min)/8*4)=3;
    quan(min+(max-min)/8*4<input&input<min+(max-min)/8*5)=4;
    quan(min+(max-min)/8*5<input&input<min+(max-min)/8*6)=5;
    quan(min+(max-min)/8*6<input&input<min+(max-min)/8*7)=6;
    quan(min+(max-min)/8*7<input)=7;
end



end