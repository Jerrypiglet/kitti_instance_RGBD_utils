clc
close all

idx = val_idxes + 1;
X = X_homo_rect(1:3, :);
figure;
scatter3(X(1, idx), X(2, idx), X(3, idx), 10, X(3, idx), '.');
% axis vis3d
axis equal;
xlabel('x');
ylabel('y');
zlabel('z');
title('X_rectified');

KRt = K * [1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0];
x_homo = KRt * X_homo_rect;
figure;
c = x_homo(3, idx);
scatter(x_homo(1, idx)./x_homo(3, idx), x_homo(2, idx)./x_homo(3, idx), 5, c, 'filled'); 
colorbar;
xlabel('x');
ylabel('y');
axis equal;
% ylim([0, 375]);
% xlim([0, 1242]);
set(gca, 'YDir','reverse');
title('x_rectified');
