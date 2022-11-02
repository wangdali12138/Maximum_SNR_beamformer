
warning('off','all');
folder = argv(){1};
nFiles = str2num(argv(){2});

sdrs = zeros(nFiles,2);
path =[ folder  '/'];
for index = 0:1:(nFiles-1)

    xs = audioread(sprintf('%s%08u.wav',path,index));

    xRef = xs(:,1)';
    xMix = xs(:,2)';
    xGev = xs(:,3)';

    SDRmix = bss_eval_sources(xMix,xRef);
    SDRgev = bss_eval_sources(xGev,xRef);

    sdrs(index+1,1) = SDRmix;
    sdrs(index+1,2) = SDRgev;
    
    disp(index);

end

mean_sdrs = mean(sdrs,1);
mean_sdr_up = mean_sdrs(:,2)-mean_sdrs(:,1);

formatSpec2 = "The improvment SDR is: %f";
sprintf(formatSpec2, mean_sdr_up)