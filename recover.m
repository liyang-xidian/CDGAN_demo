%=============================加窗傅立叶反变换=========================
%dt 滑窗长度
%ov 重叠比例
%tfy 时频谱矩阵
%A=size(tfy,1) 时频谱帧数
function [vv]=recover(dt,ov,tfy,A)
dl=dt*(1-ov);
for q=1:A
    uuu(1,:)=real(ifft(tfy(q,:)));
    uu(1,:)=uuu(1:dt);

    if q==1
       vv=uu;
    else
       [B,C]=size(vv);
       vv=[vv,zeros(B,dl)];
       yy=[zeros(B,C-dt+dl),uu];
       vv=vv+yy;
    end
end
end