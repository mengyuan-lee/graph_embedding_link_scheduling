function quan = quantization32(input,min,max)

quan = input;

quan(input<min+(max-min)/32)=0;
quan(min+(max-min)/32<input&input<min+(max-min)/32*2)=1;
quan(min+(max-min)/32*2<input&input<min+(max-min)/32*3)=2;
quan(min+(max-min)/32*3<input&input<min+(max-min)/32*4)=3;
quan(min+(max-min)/32*4<input&input<min+(max-min)/32*5)=4;
quan(min+(max-min)/32*5<input&input<min+(max-min)/32*6)=5;
quan(min+(max-min)/32*6<input&input<min+(max-min)/32*7)=6;
quan(min+(max-min)/32*7<input&input<min+(max-min)/32*8)=7;
quan(min+(max-min)/32*8<input&input<min+(max-min)/32*9)=8;
quan(min+(max-min)/32*9<input&input<min+(max-min)/32*10)=9;
quan(min+(max-min)/32*10<input&input<min+(max-min)/32*11)=10;
quan(min+(max-min)/32*11<input&input<min+(max-min)/32*12)=11;
quan(min+(max-min)/32*12<input&input<min+(max-min)/32*13)=12;
quan(min+(max-min)/32*13<input&input<min+(max-min)/32*14)=13;
quan(min+(max-min)/32*14<input&input<min+(max-min)/32*15)=14;
quan(min+(max-min)/32*15<input&input<min+(max-min)/32*16)=15;
quan(min+(max-min)/32*16<input&input<min+(max-min)/32*17)=16;
quan(min+(max-min)/32*17<input&input<min+(max-min)/32*18)=17;
quan(min+(max-min)/32*18<input&input<min+(max-min)/32*19)=18;
quan(min+(max-min)/32*19<input&input<min+(max-min)/32*20)=19;
quan(min+(max-min)/32*20<input&input<min+(max-min)/32*21)=20;
quan(min+(max-min)/32*21<input&input<min+(max-min)/32*22)=21;
quan(min+(max-min)/32*22<input&input<min+(max-min)/32*23)=22;
quan(min+(max-min)/32*23<input&input<min+(max-min)/32*24)=23;
quan(min+(max-min)/32*24<input&input<min+(max-min)/32*25)=24;
quan(min+(max-min)/32*25<input&input<min+(max-min)/32*26)=25;
quan(min+(max-min)/32*26<input&input<min+(max-min)/32*27)=26;
quan(min+(max-min)/32*27<input&input<min+(max-min)/32*28)=27;
quan(min+(max-min)/32*28<input&input<min+(max-min)/32*29)=28;
quan(min+(max-min)/32*29<input&input<min+(max-min)/32*30)=29;
quan(min+(max-min)/32*30<input&input<min+(max-min)/32*31)=30;
quan(min+(max-min)/32*31<input)=31;


end