% HOEK BEREKENEN BIJ GEGEVEN AFSTAND
function [theta_graden] = testhoekberekening(dist)
%alle mogelijke hoeken
    r = 0.001;
    rho = 1000;
    C = 0.30;
    K = (r^2)*pi*rho*C;
    g = 9.81;
    v0 = 14;
    x0 = 0;
    y0 = 2.5;
    tspan = [0:0.005:0.8];
    theta = -38:0.1:0;
    D = zeros(length(theta), 1);

    f = @(t,N) [N(3); N(4);-K*N(3)*norm([N(3),N(4)]); 
                       -g-K*N(4)*norm([N(3),N(4)])];
    
%over alle hoeken de afstanden berekenen en dichtstbij de gewilde afstand
    for i = 1:length(theta)
        B = [x0,y0,v0*cosd(-38.1+i/10),v0*sind(-38.1+i/10)];
        %options = odeset('Events',@eventfunction,'RelTol',1e-6);
        [~, N] = ode45(f, tspan, B);

        D(i) = min((N(:,1)-dist).^2+((N(:,2)).^2));
    end
    error = sqrt(min(D));
    [~, idx] = min(D);
    theta_graden = theta(idx);
    
    %B_best = [x0,y0,v0*cosd(theta_graden),v0*sind(theta_graden)];
    %options = odeset('Events',@eventfunction,'RelTol',1e-6);
    %[t,M] = ode45(f, tspan, B_best);
    %hold on
    %plot(M(:,1),M(:,2))
    %t = xlabel('afstand (m)');
    %ylabel('hoogte (m)')
    %xL = xlim;
    %yL = ylim; 
    %line(xL, [0 0]);  
    %hold off