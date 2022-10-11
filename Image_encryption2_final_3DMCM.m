% refresh memory, command window and variables...
clear, clc, close all

% read the image I have put this resize to enable you to change image size,
% just be careful of the dimensions;
I = imresize(imread('cameraman.tif'), [64, 64]);
[M, N, ~] = size(I);

% Defining m which is the block size in the paper, the condition is that
% MxN = m^3..
% e.g. 512x512 image goes with 64x64x64 blocks (m = 64)
% .... 64x64 image goes with 16x16x16 blocks (m = 16)
m = 16;

% The method starts with converting the image to 3d image consisting of
% [16-16-by-16] pathches, (This what the paper uses). 
% The paper uses 64x64x64 but  found it too slow to perform


I3d = im3d(I, [m, m]);
% I3d = im3d(I, [64, 64]);


% There was no information in  the paper regarding generating this key as
% the Arnold map and the 3DMCM map are used in the later stages of the
% paper, however, the dynamic range was shown to be 2^8, so I used the rand
% function for this key
rng(1)
key_space_range = (2^8 - 1);
keypreD = round(rand([m, m]) * key_space_range);


% Z_bar is the name used in paper for this step, I also used it here so you
% can easily follow up, please read the help of the function for more
% information
disp('====================== Start of the encoding phase ===================')
disp('Pre-Diffusion encoding step')
Z_bar = pre_diffuse_encode(double(I3d), keypreD);

% Shuffling the 3D image Z_bar using the 3DMCM coding.
% This time the shuffling will be done in the 3D image (Z_bar) once, please
% refer to the paper (section 4.2)
disp('Chaotic map permution encoding of the extracted planes')
disp('this may take some time depending on the image size')

x0_y0 = [0.1, 0.1, 0.1]';
A = [1, 1, 1; 1, 1, 2; 56, 21, 19];
perm_ind = my_3DMCM_enc(16, m, A, x0_y0);

% Shuffling the image, the reshape to guarntee the image shape will be
% preserved after the permutation
Z_bar_enc = reshape(Z_bar(perm_ind), [m, m, m]);

% After performing the 3DMCM it's time now to permute the bit planes. 
% *Note, thiss time the image is already shuffled this will only be
% altering the planes positions
Z_bar_im = inv_im3d(Z_bar_enc, [m, m], [M, N]);

disp('Extraction of the planes')
n_planes = 8;
bin_I = zeros(M, N, n_planes);

for i = 1:n_planes
   bin_I(:, :, i) = bitget(Z_bar_im, i);
end

% Extraction of the planes
bin_I_0_7 = cat(3, bin_I(:, :, 1),  bin_I(:, :, 8));
bin_I_1_6 = cat(3, bin_I(:, :, 2),  bin_I(:, :, 7));
bin_I_2_5 = cat(3, bin_I(:, :, 3),  bin_I(:, :, 6));
bin_I_3_4 = cat(3, bin_I(:, :, 4),  bin_I(:, :, 5));


% Combining the the shuffled planes to form the permuted encrypted image...
I_bin_perm = cat(3, bin_I_0_7, bin_I_1_6, bin_I_2_5, bin_I_3_4);

% Converting the plane back to decimal from the binary form
Z2_enc = uint8(my_bin2dec(I_bin_perm));

% Final step of the encoding step is to perform the post diffusion step
% which is done through the left circular shift
disp('Performing post permutation encoding with circular shift')
rng(1)
key_space_range = (2^8 - 1);
PostkeyD = round(rand([256, 256]) * key_space_range);
Z2_final_enc = post_diffusion_encode(Z2_enc, PostkeyD);

%=============End of the encrypyion phase===================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%

% This time I left some psnr(xx, yy) lines intentionally so you get to
% follow the decryption steps, these lines correponds to each image in the
% encoding step, 
% **(Inf) psnr value means that the images are identical
% ** you need of course to uncomment the lines.
% 
% Start the decoding phase from the end of the encryption phase and going
% babkward. The decoding phase starts with  the inverse of the post
% diffusion. It must use the SAME key as the post. 
% ** This is equivalnt to Z2_enc in the encoding step.
disp('===========Start of the decryption phase ==================')
disp('Post permutation decoding')
Z2_final_dec = uint8(post_diffusion_decode(Z2_final_enc, PostkeyD));
% psnr(Z2_enc, Z2_final_dec)


bin_I_dec = zeros(M, N, n_planes);
for i = 1:n_planes
   bin_I_dec(:, :, i) = bitget(uint8(Z2_final_dec), i);
end

% Extraction of the planes
disp('Planes extration')
bin_I_0_7_dec = cat(3, bin_I_dec(:, :, 8),  bin_I_dec(:, :, 7));
bin_I_1_6_dec = cat(3, bin_I_dec(:, :, 6),  bin_I_dec(:, :, 5));
bin_I_2_5_dec = cat(3, bin_I_dec(:, :, 4),  bin_I_dec(:, :, 3));
bin_I_3_4_dec = cat(3, bin_I_dec(:, :, 2),  bin_I_dec(:, :, 1));
% psnr(bin_I_0_7_dec, bin_I_0_7)
% psnr(bin_I_1_6_dec, bin_I_1_6)
% psnr(bin_I_2_5_dec, bin_I_2_5)
% psnr(bin_I_3_4_dec, bin_I_3_4)


% Combining the the shuffled planes to form the permuted encrypted image...
I_bin_dec = cat(3, bin_I_0_7_dec(:, :, 1), bin_I_1_6_dec(:, :, 1), ...
    bin_I_2_5_dec(:, :, 1), bin_I_3_4_dec(:, :, 1), bin_I_3_4_dec(:, :, 2), ...
    bin_I_2_5_dec(:, :, 2), bin_I_1_6_dec(:, :, 2), bin_I_0_7_dec(:, :, 2));


% Converting the plane back to decimal from the binary form Note that a new
% function called my_bin2dec2 this time, because the planes are reverted,
% i.e. plane 1 on I_bin_dec is equivalent to plane 8 in I_bin in the
% encoding step.
Z2_dec = uint8(my_bin2dec2(I_bin_dec));
% psnr(double(bin_I), double(I_bin_dec))
% psnr(double(Z2_dec), Z_bar_im)

%---------------
% Now inverting the 3DMCM step, firstly we need to convert the image back
% to 3D to be able to perform the 3DMCM.
disp('Decoding of the planes using the inverse of the chaotic maps (invert 3DMCM)')
Z2_dec_3D = im3d(Z2_dec, [m, m]);

% Then, we need to extract the 3D retreiving indices
x0_y0 = [0.1, 0.1, 0.1]';
A = [1, 1, 1; 1, 1, 2; 56, 21, 19];
perm_dec = my_3DMCM_dec(16, m, A, x0_y0);

% Again the reshape to guarentee the image to be 3D after the shffling,
% note it's [m, m, m]
Z_bar_dec = double(reshape(Z2_dec_3D(perm_dec), [m, m, m]));
% psnr(Z_bar_dec, Z_bar)

% Recovering the pre-deffused to obtain the final recovered image
disp('Finally, performing the pre-Diffusion decoding step')
I3d_dec = pre_diffuse_decode(Z_bar_dec, keypreD);

% Finally, retreiving the original image by getting t
dec_final_image = inv_im3d(I3d_dec, [m, m], [M, N]);
% psnr(double(I), dec_final_image)


%%
close all

% Encoding phase images
figure, imshow(I), title('Original image')
figure, imshow(Z_bar_im, []), impixelinfo, title('Pre-diffusion encoded image')
figure, imshow(Z2_enc), impixelinfo, title('3DMCM permuted image')
figure, imshow(Z2_final_enc, []), title('Post-diffusion encoded image (Final Encoded image)')

% Decoding phase images
figure, imshow(Z2_final_dec, []), title('Post-diffusion decoded image')
figure, imshow(Z2_dec, []), impixelinfo, title('3DMCM inverse permuted image')
figure, imshow(dec_final_image, []), impixelinfo, 
title('Pre-diffusion decoded image (Final decoded image)')
%%
%========================= Support functions =============================
%=========================================================================
%=========================================================================
%=========================================================================

function out = im3d(I, sz)
% This helper function converts the image from M*N to smaller 3d mxnxz, so
% as to perform the rest of the computations on its

% It basically  uses IM2COL command which converts the image to sz(1)*sz(2)
% patches, then the reshape to concatinate them to [sz(1) by sz(2)]
out = reshape(im2col(I, sz, 'distinct'), [sz, numel(I)/(sz(1)*sz(2))]);
end

function Z_bar = pre_diffuse_encode(I3d, keypreD)
% This helper function performs the preDiffusion encosing step in the
% paper. The first image of the 3D plane is encided using the preDiffsion
% key, whereas the rest of the images are encoded using the previously
% image patch 


Z_bar = zeros(size(I3d), 'double');
Z_bar(:, :, 1) = bitxor(I3d(:, :, 1), keypreD);

for i = 2:size(I3d, 3)
    Z_bar(:, :, i) = bitxor(Z_bar(:, :, i-1), I3d(:, :, i));
end
end

function out = inv_im3d(I3d, sz, imsz)
% This helper function converts the small 3d mxnxz image back to M*N 

% It basically  uses the inverse of  IM2COL command which is COL2IM to
% convert the image back to its original dimension, prior to that it uses
% the reshape command to get back the original image

temp = reshape(I3d, [prod(sz), (prod(imsz)/prod(sz))]);
out = col2im(temp, sz, imsz, 'distinct');
end

function index2 = my_3DMCM_enc(Modulo, m, A, x0_y0)

% This function returns 3DCMC shuffling indices that can be used to encode
% an image
[x, y, z] = chaotic_map_3DMCM(Modulo, m, A, x0_y0);

[~, x3] = sort(x, 'ascend');
[~, y3] = sort(y, 'ascend');
[~, z3] = sort(z, 'ascend');

index = sub2ind([m^3, m^3, m^3], x3, y3, z3);
[~, index2] = sort(index, 'ascend');
end

function inv_index = my_3DMCM_dec(Modulo, m, A, x0_y0)
% The function generates the same shuffling indices same as encoding then
% it inverts it by resorting the data again to retrieve the original
% indices

index2 = my_3DMCM_enc(Modulo, m, A, x0_y0);
[~, inv_index] = sort(index2);
end


function [x, y, z] = chaotic_map_3DMCM(Modulo, m, A, x0_y0)
% This fuction performs the generation of the chotic maps with wither the
% Arnold Cat Maps, the 3DMCM and the 3DMCM depending on the values of the A
% matrix and the size of it.
% it takes the modulo value, the size of the image/patch (m), the A matrix
% and finialy the initial starting points x0_y0 which are fed as one column
% vector

x_y_z = [];
xt_yt = x0_y0;

for j = 1:m
    for k = 1:m
        for z = 1:m
            xt_yt = mod((A * xt_yt), Modulo);
            x_y_z = [x_y_z, xt_yt]; %#ok<AGROW>
        end
    end
end

x = x_y_z(1, :);
y = x_y_z(2, :);
z = x_y_z(3, :);
end


function perm_I = post_diffusion_encode(I, PostkeyD)

n_planes = 8;

[m, n, ~] = size(I);
% fprintf('Converting image to binary\n');
bin_I = zeros(m, n, n_planes);

for i = 1:n_planes
   bin_I(:, :, i) = bitget(I, i); 
end

bin_I2 = char(zeros([m*n, n_planes]));

ind = 1;
for i = 1:m
    for j = 1:n
        
        tmp_I = num2str(fliplr(reshape(bin_I(i, j, :), [1, n_planes])));
        bin_I2(ind, :) = strrep(tmp_I, ' ', '');
        ind = ind + 1;
    end
end

% performaing the circular shift

perm_I = zeros(size(I));

for i = 1:numel(I)
    I_c1 = circshift(bin_I2(i, :), PostkeyD(i));

    perm_I(i) = mvl2dec(I_c1);
%     if mod(i, round(numel(I)/10)) == 0
%         fprintf('Permuting %s %%\n', num2str(round(i / numel(I) * 100)));
%     end
end
perm_I = perm_I';
end

function perm_I = post_diffusion_decode(I, PostkeyD)

n_planes = 8;

[m, n, ~] = size(I);
% fprintf('Converting image to binary\n');
bin_I = zeros(m, n, n_planes);

for i = 1:n_planes
   bin_I(:, :, i) = bitget(I, i); 
end

bin_I2 = char(zeros([m*n, n_planes]));

ind = 1;
for i = 1:m
    for j = 1:n
        tmp_I = num2str(fliplr(reshape(bin_I(i, j, :), [1, n_planes])));
        bin_I2(ind, :) = strrep(tmp_I, ' ', '');
        ind = ind + 1;
    end
end

% performaing the circular shift

perm_I = zeros(size(I));

for i = 1:numel(I)
    I_c1 = circshift(bin_I2(i, :), -PostkeyD(i));

    perm_I(i) = mvl2dec(I_c1);
%     if mod(i, round(numel(I)/10)) == 0
%         fprintf('Permuting %s %%\n', num2str(round(i / numel(I) * 100)));
%     end
end
perm_I = perm_I';
end

function res = my_bin2dec(a)


n_planes = size(a, 3);
pow_array = reshape(2.^(n_planes-1:-1:0), [1, 1, n_planes]);
res = sum(bsxfun(@times, a, pow_array), 3);
end


function res = my_bin2dec2(a)


n_planes = size(a, 3);
pow_array = reshape(2.^(0:n_planes-1), [1, 1, n_planes]);
res = sum(bsxfun(@times, a, pow_array), 3);
end


function I3d_dec = pre_diffuse_decode(Z_bar, keypreD)

% This helper function performs the preDiffusion decoding step in the
% paper. It takes the ciphered 3D patches and resores the plane 3D image
% patches back from it

I3d_dec = zeros(size(Z_bar), 'double');
I3d_dec(:, :, 1) = bitxor(Z_bar(:, :, 1), keypreD);

for i = 2:size(Z_bar, 3)
    I3d_dec(:, :, i) = bitxor(Z_bar(:, :, i-1), Z_bar(:, :, i));
end
end
