%=============================�Ӵ�����Ҷ���任=========================
%dt ��������
%ov �ص�����
%tfy ʱƵ�׾���
%A=size(tfy,1) ʱƵ��֡��
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