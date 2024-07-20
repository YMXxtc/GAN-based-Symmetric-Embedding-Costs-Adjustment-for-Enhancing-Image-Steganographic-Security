clear;
close all;
clc;
payload = 0.4;

coverPath = '/data1/ymx/BOSSBase/BB-cover-resample-256-testset';   %/data1/ymx/BOSSBase/BossBase-1.01-cover-resample-256

stegoPath = sprintf('/data1/ymx/HILL_bb/%.1f',payload);

if ~exist(stegoPath,'dir')
    mkdir(stegoPath);
end
%tic
for i = 1:10000
    path = sprintf('%s/%d.pgm',coverPath,i);
    cover = imread(path);
    cover = double(cover);

    %R = load(sprintf('%s/%d.mat',RPath,i)).R;

    [stego, p, modify, costP, costM, rhoB] = HILL_modified(cover,payload);
    
%    figure; imshow(uint8(stego));
%    figure; imshow(p);
%    figure; imshow(costP);
%    figure; imshow(costM);
    imwrite(uint8(stego),sprintf('%s/%d.pgm',stegoPath,i));
%     imwrite(uint8(stego),sprintf('%s/%d.png',stegoPath,i));
%    imwrite(p,sprintf('%s/%d_P.png',stegoPath,i));
    imwrite(modify,sprintf('%s/%d_M.png',stegoPath,i));
%     imwrite(costP,sprintf('%s/%d_rhoP.png',stegoPath,i)); imwrite(costM,sprintf('%s/%d_rhoM.png',stegoPath,i));

    save(sprintf('%s/%d_p.mat',stegoPath,i), 'p');
    save(sprintf('%s/%d_rhoP.mat',stegoPath,i), 'costP');
    save(sprintf('%s/%d_rhoM.mat',stegoPath,i), 'costM');
    % save(sprintf('%s/%d_rhoB.mat',stegoPath,i), 'rhoB');
end
%toc
