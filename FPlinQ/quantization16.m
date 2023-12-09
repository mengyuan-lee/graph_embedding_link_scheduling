function quan = quantization16(input,min,max)

quan = input;

quan(input<min+(max-min)/16)=0;
quan(min+(max-min)/16<input&input<min+(max-min)/16*2)=1;
quan(min+(max-min)/16*2<input&input<min+(max-min)/16*3)=2;
quan(min+(max-min)/16*3<input&input<min+(max-min)/16*4)=3;
quan(min+(max-min)/16*4<input&input<min+(max-min)/16*5)=4;
quan(min+(max-min)/16*5<input&input<min+(max-min)/16*6)=5;
quan(min+(max-min)/16*6<input&input<min+(max-min)/16*7)=6;
quan(min+(max-min)/16*7<input&input<min+(max-min)/16*8)=7;
quan(min+(max-min)/16*8<input&input<min+(max-min)/16*9)=8;
quan(min+(max-min)/16*9<input&input<min+(max-min)/16*10)=9;
quan(min+(max-min)/16*10<input&input<min+(max-min)/16*11)=10;
quan(min+(max-min)/16*11<input&input<min+(max-min)/16*12)=11;
quan(min+(max-min)/16*12<input&input<min+(max-min)/16*13)=12;
quan(min+(max-min)/16*13<input&input<min+(max-min)/16*14)=13;
quan(min+(max-min)/16*14<input&input<min+(max-min)/16*15)=14;
quan(min+(max-min)/16*15<input)=15;


end