figure(1);
img = imread('iris_grey_IR.png');
subplot(1,3,1);
imshow(img);

darkRimRadius = 145;
darkModulationAmplitude = 0.5;
A_e = 1 / darkModulationAmplitude - 1;
innerWidth = 30;
outerWidth = 10;
[xx,yy] = meshgrid(-505:518,-513:510);
rr = sqrt(xx.^2 + yy.^2) - darkRimRadius;
modulation_inner_rr = 1 ./ (1 + A_e * exp( -(abs(rr) / innerWidth).^2));
modulation_outer_rr = 1 ./ (1 + A_e * exp( -(abs(rr) / outerWidth).^2));
modulation = nan(size(img));
modulation(rr <= 0) = modulation_inner_rr(rr <= 0);
modulation(rr > 0) = modulation_outer_rr(rr > 0);
modulation = 1 - 2 * (modulation - 0.5);
subplot(1,3,2);
% mesh(modulation);
plot(modulation(512,:));
imwrite(uint8(modulation * 255),'iris_modulation.png');

subplot(1,3,3);
darkened_img = uint8(double(img).*modulation);
imshow(darkened_img);
imwrite(darkened_img,'iris_grey_IR_boundary_darkened.png');