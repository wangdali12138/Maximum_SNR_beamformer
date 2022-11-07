warning('off','all');
inputfolderdir = 'C:\Users\ASUS\Desktop\streetnet\test_max_ReSpeaker_USB_2\';
outputfolderdir = 'C:\Users\ASUS\Desktop\streetnet\test_max_ReSpeaker_USB_2\';
nFiles = 50;
fs = 16000;

for index = 0:1:(nFiles-1)

    xs = audioread(sprintf('%s%08u.wav',inputfolderdir,index));

    xRef = xs(:,1)';
    xMix = xs(:,2)';
    xGev = xs(:,3)';
    
    audiowrite(sprintf('%s%08u%s.wav',outputfolderdir,index,'Ref'),xRef,fs);
    audiowrite(sprintf('%s%08u%s.wav',outputfolderdir,index,'Gev'),xGev,fs);
    audiowrite(sprintf('%s%08u%s.wav',outputfolderdir,index,'Mix'),xMix,fs);
    disp(index);
end