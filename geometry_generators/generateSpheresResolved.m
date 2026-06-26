function sphere_data_resolved = generateSpheresResolved(num_target_spheres, min_r, max_r, outer_radius, min_separation_distance)
% Generates a set of non-intersecting polydisperse spheres confined inside a
% spherical container of radius outer_radius (centered at the origin).
%
% Inputs:
%   num_target_spheres      - The desired number of spheres to generate.
%   min_r                   - Minimum radius for an interior sphere.
%   max_r                   - Maximum radius for an interior sphere.
%   outer_radius            - Radius of the spherical container.
%   min_separation_distance - Minimum gap between sphere surfaces (and between
%                             a sphere surface and the container wall).
%
% Output:
%   sphere_data_resolved - An N x 4 matrix where N is the number of successfully placed spheres.
%                          Each row is [x, y, z, r] representing center coordinates and radius.

    sphere_data_resolved = []; % Initialize empty matrix for spheres
    max_attempts_per_sphere = 5000; % Max attempts to place a single sphere

    fprintf('Attempting to generate %d non-intersecting spheres in a sphere of radius %.4f...\n', ...
            num_target_spheres, outer_radius);

    for i = 1:num_target_spheres
        found_position = false;
        current_sphere_attempts = 0;

        while ~found_position && current_sphere_attempts < max_attempts_per_sphere
            % 1. Generate a random radius
            r_new = min_r + (max_r - min_r) * rand();

            % 2. Generate a random center uniformly inside the ball of radius
            %    (outer_radius - r_new - min_separation_distance) so the new
            %    sphere stays fully inside the container, clear of the wall.
            R_place = outer_radius - r_new - min_separation_distance;
            if R_place <= 0
                current_sphere_attempts = current_sphere_attempts + 1;
                continue; % sphere too big to fit in the container
            end
            % Uniform-in-ball sampling: random direction * radius scaled by rand^(1/3)
            dir = randn(1, 3);
            dir = dir / norm(dir);
            radial = R_place * rand()^(1/3);
            center_new = radial * dir;

            intersects = false;

            % 3. Check for intersection with all previously placed spheres
            for j = 1:size(sphere_data_resolved, 1)
                center_existing = sphere_data_resolved(j, 1:3);
                r_existing = sphere_data_resolved(j, 4);

                % Plain Euclidean distance between centers (no periodicity)
                dist_centers = norm(center_new - center_existing);

                % Sum of radii plus required gap
                sum_radii = r_new + r_existing + min_separation_distance;

                % Check for intersection
                if dist_centers < sum_radii
                    intersects = true;
                    break; % Intersection found, break from inner loop and try a new random sphere
                end
            end

            % 4. If no intersection, add the new sphere to the list
            if ~intersects
                sphere_data_resolved = [sphere_data_resolved; center_new, r_new];
                found_position = true;
            end
            current_sphere_attempts = current_sphere_attempts + 1;
        end

        if ~found_position
            fprintf('Warning: Could not place sphere %d after %d attempts.\n', i, max_attempts_per_sphere);
            fprintf('         Generation stopped at %d spheres. This usually means the packing limit is reached.\n', size(sphere_data_resolved, 1));
            break; % Stop trying to add more spheres if one cannot be placed
        end
    end
    fprintf('Successfully generated %d non-intersecting spheres.\n', size(sphere_data_resolved, 1));
end
