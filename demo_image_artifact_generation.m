%% Load Image
img = single(imread('lenna.png'))/255.0;

% gray image generation
% img = mean(img, 3);

sz = size(img);

if length(sz) == 2
    cmap = 'gray';
    sz(3) = 1;
else
    cmap = [];
end

figure(1);
colormap(cmap);

imagesc(img, [0, 1]);
axis image off;
title('Ground Truth');

%% 1-1. Inpainting: uniform sampling
ds_y = 2;
ds_x = 4;

msk = zeros(sz);
msk(1:ds_y:end, 1:ds_x:end) = 1;

dst = img .* msk;

figure(2);
colormap(cmap);

subplot(131);
imagesc(img, [0, 1]);
axis image off;
title('Ground Truth');

subplot(132);
imagesc(msk, [0, 1]);
axis image off;
title('Uniform sampling Mask');

subplot(133);
imagesc(dst, [0, 1]);
axis image off;
title('Sampling Image');

%% 1-2. Inpainting: random sampling
%rnd = rand(sz);
%prob = 0.5;
%msk = single(rnd < prob);

rnd = rand(sz(1:2));
prob = 0.5;
msk = single(rnd < prob);
msk = repmat(msk, 1, 1, sz(3));

dst = img .* msk;

figure(3);
colormap(cmap);

subplot(131);
imagesc(img, [0, 1]);
axis image off;
title('Ground Truth');

subplot(132);
imagesc(msk, [0, 1]);
axis image off;
title('Random Sampling Mask');

subplot(133);
imagesc(dst, [0, 1]);
axis image off;
title('Sampled Image');

%% 1-3. Inpainting: gaussian sampling
ly = linspace(-1, 1, sz(1));
lx = linspace(-1, 1, sz(2));

[mx, my] = meshgrid(lx, ly);

mu_x = 0;
mu_y = 0;
sgm_x = 1;
sgm_y = 1;

wgt = 1.0;

% gaus = wgt * exp(-(((mx - mu_x).^2) ./ (2 * sgm_x^2) + ((my - mu_y).^2) ./ (2 * sgm_y^2)));
% gaus = repmat(gaus, 1, 1, sz(3));
% rnd = rand(sz);
% msk = single(rnd < gaus);

gaus = wgt * exp(-(((mx - mu_x).^2) ./ (2 * sgm_x^2) + ((my - mu_y).^2) ./ (2 * sgm_y^2)));
gaus = repmat(gaus, 1, 1, 1);
rnd = rand(sz(1:2));
msk = single(rnd < gaus);
msk = repmat(msk, 1, 1, sz(3));

dst = img .* msk;

figure(4);
colormap(cmap);

subplot(131);
imagesc(img, [0, 1]);
axis image off;
title('Ground Truth');

subplot(132);
imagesc(msk, [0, 1]);
axis image off;
title('Gaussian Sampling Mask');

subplot(133);
imagesc(dst, [0, 1]);
axis image off;
title('Sampled Image');

%% 2-1. Denosinge: Random noise
sgm = 60.0;

noise = sgm / 255.0 * randn(sz);

dst = img + noise;

figure(5);
colormap(cmap);

subplot(131);
imagesc(img, [0, 1]);
axis image off;
title('Ground Truth');

subplot(132);
imagesc(noise, [0, 1]);
axis image off;
title('Random Noise');

subplot(133);
imagesc(dst, [0, 1]);
axis image off;
title(['Noisy Image with ' num2str(sgm, '%.2f sigma')]);

%% 2-2. Denoising: poisson noise (image)
dst = poissrnd(255.0 * img) / 255.0;
noise = dst - img;

figure(6);
colormap(cmap);

subplot(131);
imagesc(img, [0, 1]);
axis image off;
title('Ground Truth');

subplot(132);
imagesc(noise, [0, 1]);
axis image off;
title('Poisson Noise');

subplot(133);
imagesc(dst, [0, 1]);
axis image off;
title('Noisy Image');

%% 2-2. Denoising: poisson noise (CT)
N = 512;
ANG = 180;
VIEW = 360;
THETA = linspace(0, ANG, VIEW + 1);
THETA = THETA(1:VIEW);

A = @(x) radon(x, THETA);
AT = @(y) iradon(y, THETA, 'none', N);
AINV = @(y) iradon(y, THETA, N);

pht = 0.03 * phantom(N);

prj = A(pht);

i0 = 1e4;
dst = exp(-prj);
dst = poissrnd(i0 * dst);
dst(dst < 1) = 1;
dst = -log(dst / i0);
dst(dst < 0) = 0;

noise = dst - prj;

rec = AINV(prj);
rec_noise = AINV(noise);
rec_dst = AINV(dst);

figure(7);
colormap('gray');

subplot(241);
imagesc(pht, [0, 0.03]);
axis image off;
title('Ground Truth');

subplot(242);
imagesc(rec, [0, 0.03]);
axis image off;
title('Reconstuction');

subplot(243);
imagesc(rec_noise);
axis image off;
title('Reconstruction using Noise');

subplot(244);
imagesc(rec_dst, [0, 0.03]);
axis image off;
title('Reconstruction using Noisy data');

subplot(246);
imagesc(prj);
title('Projection data');

subplot(247);
imagesc(noise);
title('Poisson Noise in Projection');

subplot(248);
imagesc(dst);
title('Noisy data');

%% 3. Super-resolution
% -----------------------
% Order option
% -----------------------
% 'nearest'    - nearest-neighbor interpolation
% 'bilinear'   - bilinear interpolation
% 'bicubic'    - cubic interpolation; the default method

dw = 1/5.0;
order = 'bilinear';

dst_dw = imresize(img, dw, order);
dst_up = imresize(dst_dw, 1/dw, order);

figure(8);
colormap(cmap);

subplot(131);
imagesc(img, [0, 1]);
axis image off;
title('Ground Truth');

subplot(132);
imagesc(dst_dw, [0, 1]);
axis image off;
title('Downscaled Image');

subplot(133);
imagesc(dst_up, [0, 1]);
axis image off;
title('Upscaled Image');
