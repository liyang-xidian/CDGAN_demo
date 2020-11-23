close all;
clc
clear
%%  load networks and reverberation filter
load('Reverberation_filter.mat');  % reverberation filter and STFT parameters
load('network.mat');               % the trained GAN and normalization parameters
%%  read test data (target speech signal and noise)
addpath('speech')
addpath('noise')
maindir=cd;
speechdir=dir(strcat(maindir,'\speech'));
noisedir=dir(strcat(maindir,'\noise'));
speechNames = {speechdir.name};
noiseNames = {noisedir.name};
r=randperm(size(noiseNames,2)-2,1)+2;
rr=randperm(size(speechNames,2)-2,1)+2;
[mic,~]=audioread(strcat(maindir,'\noise\',noisedir(r).name));
[mic1,~]=audioread(strcat(maindir,'\speech\',speechdir(rr).name));
%%  convolutive mixture of source signals 
SN = 2;         % the number of sources
L = 150*dt;
mic2=downsample(mic1,2);
mic3=mic2(1+2000:L+2000,1).';         % delete invalide data
r=randperm(200,1);
v=0.2*mic(1+r:L+r,1).';
s=[mic3;v];

X = zeros(SN,L);
   for m = 1 : SN
       for n = 1 : SN
           hmn = squeeze(H(m,n,:));
           X(m,:) = X(m,:) + filter(hmn,1,s(n,:));%Convolutive mixing
       end
   end
x1=X(1,:);
%x2=X(2,:);
xx=mic3+v;%Additive mixing
[fx1]=STFTN(dt,ov,nfft,x1);
%[fx2]=STFTN(dt,ov,nfft,x2);
[fs]=STFTN(dt,ov,nfft,mic3);
[fxx]=STFTN(dt,ov,nfft,xx);
[fv]=STFTN(dt,ov,nfft,v);
A=size(fx1,1);
trmicxx = zeros(A-4, 2*nfft);
for p=1:A-4
    trmicxx(p,:)=[fxx(p,1:nfft/2) fxx(p+1,1:nfft/2) fxx(p+2,1:nfft/2) fxx(p+3,1:nfft/2)];
end
trmics(1:(A-4),:)=fs(3:A-2,1:nfft/2);
trmicv(1:(A-4),:)=fv(3:A-2,1:nfft/2);
angxx(1:(A-4),:)=fxx(3:A-2,1:nfft/2);
angx(1:(A-4),:)=fx1(3:A-2,1:nfft/2);
%% reverberation suppression and speech enhancement
tinput=[real(angx) imag(angx)].';
tinputn = mapminmax('apply',tinput,XPS);
anE=sim(netGXY,tinputn);
an=anE;
fin_output=an.';
tfy=fin_output(:,1:nfft/2)+1i * fin_output(:,nfft/2+1:nfft);
tfy1=[tfy tfy(:,1)];
tfy2=[tfy1,fliplr(conj(tfy(:,2:nfft/2)))];
[xxx]=recover(dt,ov,tfy2,size(tfy2,1));
A=A-8;
outxx=anE.';
outxx1=outxx(:,1:nfft/2)+1i*outxx(:,nfft/2+1:nfft);
ang=angle(outxx1(3:A+2,:));
xx1=zeros(A,2*nfft);
   for p=1:A
       xx1(p,:)=[outxx1(p,:) outxx1(p+1,:) outxx1(p+2,:) outxx1(p+3,:)];        
   end
xx1=abs(xx1.');
xxn = mapminmax('apply',xx1,EPS);
anS=sim(netGYS,xxn);
an=anS;
fin_output=an.';
tfy=fin_output.*cos(ang)+1i*fin_output.*sin(ang);
tfy1=[tfy tfy(:,1)];
tfy2=[tfy1,fliplr(conj(tfy(:,2:nfft/2)))];
[fy]=recover(dt,ov,tfy2,size(tfy2,1));
ttfy=tfy/norm(tfy,'fro');
ttfs=trmics(3:A+2,:)/norm(trmics(3:A+2,:),'fro');
fSIR=ttfy-ttfs;
SIR=20*log10(norm(ttfs,'fro')/norm(fSIR,'fro'));
%% display signal waveforms
%-----------------plot target speech signal and interference--------------
Fs = 8000;
tt=(1/Fs:1/Fs:size(mic3,2)/Fs);
figure(1);
subplot(4,1,1)
plot(tt,mic3), axis([0 1.2 -0.2 0.2]);
xlabel('Time');
ylabel('Amplitude');
title('Target speech');
subplot(4,1,2)
spectrogram(mic3.',dt,ov*dt,128,Fs,'yaxis');
subplot(4,1,3)
plot(tt,v), axis([0 1.2 -0.2 0.2]);
xlabel('Time');
ylabel('Amplitude');
title('interference');
subplot(4,1,4)
spectrogram(v.',dt,ov*dt,128,Fs,'yaxis');
%--------------------------------end---------------------------------------

%-----------------plot convolutive mixture and additive mixture------------
figure(2);
subplot(2,1,1)
plot(tt,x1), axis([0 1.2 -0.4 0.4]);
xlabel('Time');
ylabel('Amplitude');
title('Convolutive mixture');
subplot(2,1,2)
spectrogram(x1.',dt,ov*dt,128,Fs,'yaxis');

figure(3);
subplot(2,1,1)
plot(tt,xx), axis([0 1.2 -0.2 0.2]);
xlabel('Time');
ylabel('Amplitude');
title('Additive mixture');
subplot(2,1,2)
spectrogram(xx.',dt,ov*dt,128,Fs,'yaxis');
%--------------------------------end---------------------------------------

%--------plot reverberation suppression and speech enhancement result------
tt=(1/Fs:1/Fs:size(xxx,2)/Fs);
figure(4);
subplot(2,1,1)
plot(tt,xxx), axis([0 1.2 -0.2 0.2]);
xlabel('Time');
ylabel('Amplitude');
title('Reverberation suppression result');
subplot(2,1,2)
spectrogram(xxx.',dt,ov*dt,128,Fs,'yaxis');
tt=(1/Fs:1/Fs:size(fy,2)/Fs); 
figure(5);
subplot(2,1,1)
plot(tt,fy), axis([0 1.2 -0.2 0.2]);
xlabel('Time');
ylabel('Amplitude');
title(['Enhancement result, SIR = ',num2str(SIR),'dB']);
subplot(2,1,2)
spectrogram(fy.',dt,ov*dt,128,Fs,'yaxis');
%--------------------------------end--------------------------------------- 
%% sub_functions
%============================signal recover=========================
%dt: Window length
%ov: Overlap ratio
%tfy: time-frequency spectrum
%A=size(tfy,1) Number of Frames
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

%=============================end=========================


%=============================STFT=========================
%dt: Window length
%ov: Overlap ratio
%nfft: Number of frequency bins
%mic: Time-domain signal
function [x]=STFTN(dt,ov,nfft,mic)
dl=dt*(1-ov);
    if ov==0
       h=ones(1,dt);
    else
       h=hamming(dt)';
    end

len = size(mic,2);
for z=0:fix((len+dl-dt)/dl)-1

    if z==0
       x=(fft(mic(1,dl*z+1:dl*z+dt).*h,nfft));
       %x2=(fft(mic(2,dl*z+1:dl*z+dt).*h));
    else
       x=[x;(fft(mic(1,dl*z+1:dl*z+dt).*h,nfft))];
       %x2=[x2;(fft(mic(2,dl*z+1:dl*z+dt).*h))];
    end
end
end
%=============================end=========================