function output = AdjustTone(input, alpha)

input(input > 1) = 1;
input(input < 0) = 0;

output = input.^(1/alpha);
output = rgb2hsv(output);
output(:, :, 2) = output(:, :, 2) * alpha;
output = hsv2rgb(output);

end
