rng(1);

% --- 1. Container + packing parameters ---
outer_radius = 1.0;            % radius of the spherical container
Nptcl = 100;                   % number of interior spheres to attempt
min_r = 0.08;                  % minimum interior-sphere radius (~0.1)
max_r = 0.12;                  % maximum interior-sphere radius (~0.1)
min_separation_distance = 0.001; % minimum surface-to-surface gap

sphere_data = generateSpheresResolved(Nptcl, min_r, max_r, outer_radius, min_separation_distance);

% --- 2. Create a new figure for plotting ---
figure;
hold on; % Allow multiple plots on the same axes

% --- 3. Plot the bounding sphere (container) first ---
[Xc, Yc, Zc] = sphere(40);
surf(Xc * outer_radius, Yc * outer_radius, Zc * outer_radius, ...
     'FaceColor', [0.8 0.8 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.1);

% --- 4. Set up the plot aesthetics ---
axis equal; % Ensures that the spheres are drawn as spheres (not distorted)
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
title('Polydisperse Spheres in a Spherical Container');
grid on; % Add a grid for better spatial reference
view(3); % 3D view

% Add lighting for better visual effect (makes spheres look more 3D)
camlight('headlight'); % Light source from camera position
lighting gouraud; % Smooth lighting

% --- 5. Plot each interior sphere ---
num_spheres = size(sphere_data, 1); % Get the number of rows (spheres)

% Generate a unit sphere (can be done once to save computation)
[X_unit, Y_unit, Z_unit] = sphere(20); % 20 is the number of faces for smoothness

tot_vol = 0;

for i = 1:num_spheres
    % Extract data for the current sphere
    center_x = sphere_data(i, 1);
    center_y = sphere_data(i, 2);
    center_z = sphere_data(i, 3);
    radius   = sphere_data(i, 4);

    tot_vol = tot_vol + 4*pi*radius^3/3;

    % Scale and translate the unit sphere to the current sphere's size and position
    X_sphere = X_unit * radius + center_x;
    Y_sphere = Y_unit * radius + center_y;
    Z_sphere = Z_unit * radius + center_z;

    % Plot the sphere
    surf(X_sphere, Y_sphere, Z_sphere, 'FaceColor', [0.5 0.5 0.5], ... % Grey color
         'EdgeColor', 'none', 'FaceAlpha', 0.8); % No edge lines, slight transparency
end

hold off; % Release the hold on the plot

container_vol = 4*pi*outer_radius^3/3;
fprintf("\n Total interior-sphere volume is %f.", tot_vol);
fprintf("\n Volume fraction (relative to container) is %f.\n", tot_vol / container_vol);

% ---- Save data to file with 8 digits precision
filename = strcat('sphere_data_', num2str(Nptcl), '_ball.txt');
fileID = fopen(filename, 'w');
fprintf(fileID, '# outer_radius = %.8f\n', outer_radius);
formatSpec = '%.8f %.8f %.8f %.8f\n';
Data = sphere_data';
fprintf(fileID, formatSpec, Data(:));
fclose(fileID);
