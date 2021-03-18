clear all; close all; clc;

%% Load in the video
v= VideoReader('ski_drop_low.mp4'); 
video = read (v) ;
frames = get(v,'numberOfFrames');
[width height rgbSize frames] = size(video);


%% Reshape each frame of the video into a column vector and form a 
%  matrix where each column is a frame. 
X = [];

for i = 1:frames
    x = rgb2gray(video(:,225:724,:,i));
    x = double(reshape(x(:,:),width*500,1));
    X= [X x];
end

%% 

X_1 = X(:, 1:end-1);
X_2 = X(:, 2:end);

[U1,S1,V1] = svd(X_1, 'econ');

%%
plot(diag(S1)/sum(diag(S1))*100, 'ko', 'Linewidth', [1]);
set(gca, 'Fontsize', 16);
xlabel('Singular Values');
ylabel('Energy of Singular Value (%)')
title("Energy of Singular Values - Skier Drop")

diagonal_S = diag(S1)/sum(diag(S1))*100;

sum_sing = 0;
i = 1;
while (sum_sing < 80.0)
    sum_sing = sum_sing + diagonal_S(i);
    i = i+1;
end 

%%


rank=i;
U=U1(:,1:rank); S =S1(1:rank,1:rank); V = V1(:,1:rank);

%%
S_tilde = U' * X_2 * V * diag(1./diag(S)); 

[eigVec, eigVal] = eig(S_tilde); 

Phi=U*eigVec;

mu=diag(eigVal);
dt = 1;
omega=log(mu)/dt;

y_0 = Phi\X_1(:,1);

%%
plot(real(omega), imag(omega), 'ko', 'Linewidth', [1]);
set(gca, 'Fontsize', 16);
xlabel("Real Part of Omega")
ylabel("Imaginary Part of Omega")
title("Plot of Omega Values - Skier")

%% 

time_step = size(X_1, 2);
t = (0:time_step-1)*dt;

bg_mode = find(abs(omega) < 0.1);
omega_bg = omega(bg_mode);
Phi_bg = Phi(:,bg_mode);
y0_bg = y_0(bg_mode);

u_modes = zeros(length(y0_bg), size(X_1, 2));

for iter = 1:time_step
    u_modes(:,iter) =(y0_bg .*exp(omega_bg*t(iter)));
end
u_dmd = Phi_bg*u_modes;   % DMD resconstruction with all modes

%%

X_s = X_1 - abs(u_dmd);
R = X_s .* (X_s < 0);

X_lr = uint8(abs(u_dmd));
X_sparse = (X_s - min(X_s)) ./ (max(X_s) - min(X_s));

dmd_reconstruction = uint8(abs(u_dmd) + R) + uint8(X_s -R);

%%
close all;
for i = 1:378
    image_low_rank(:,:,i) = (reshape(X_lr(:,i),width,500));
end

for i = 1:378
    imshow(image_low_rank(:,:,i));
    i
    drawnow
end 

%%
close all;
for i = 1:378
    image_sparse(:,:,i) = (reshape(X_sparse(:,i),width,500));
end

for i = 1:378
    imshow((image_sparse(:,:,i)));
    i
    drawnow
end 

%%
figure(1)
subplot(1,2,1);
imshow(image_low_rank(:,:,100))
title("Frame 100 of Background Video");

subplot(1,2,2);
imshow(image_sparse(:,:,100))
title("Frame 100 of Foreground Video");

figure(2)
subplot(1,2,1);
imshow(rgb2gray(video(:,225:724,:,100)))
title("Frame 100 of Original Video");

subplot(1,2,2)
imshow(reshape(dmd_reconstruction(:,100),width,500));
title("DMD Reconstruction")

