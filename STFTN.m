%=============================加窗傅立叶变换=========================
%dt 滑窗长度
%ov 重叠比例
%nfft 频点长度
%mic 时域信号
function [x]=STFTN(dt,ov,nfft,mic)
dl=dt*(1-ov);%滑动距离=帧长度-重叠长度
if ov==0
    h=ones(1,dt);
else
h=hamming(dt)';
end

len=size(mic,2);
for z=0:fix((len+dl-dt)/dl)-1
%Fs=5000; % Sampling frequency 

%for k=1:SN
%    if k==1
%       fftsounds=real(fft(mic(k,dt*z+1:dt*z+dt),Fs));
%       size(fftsounds)
%    else
%       fftsounds=[fftsounds;real(fft(mic(k,dt*z+1:dt*z+dt)e,Fs))];
%    end
     
%f=[1:Fs/2];
%figure(2)
%subplot(SNS,2,k)
%plot(f,fftsounds(k,f)), axis([0 4000 -15 15]);
%title('Frequency spectra of the microphone')
%xlabel('Frequency (Hz)');
%end

    if z==0
       x=(fft(mic(1,dl*z+1:dl*z+dt).*h,nfft));
       %x2=(fft(mic(2,dl*z+1:dl*z+dt).*h));
    else
       x=[x;(fft(mic(1,dl*z+1:dl*z+dt).*h,nfft))];
       %x2=[x2;(fft(mic(2,dl*z+1:dl*z+dt).*h))];
    end
end