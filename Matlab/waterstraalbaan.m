% PLOTTEN VAN BAAN VAN WATERSTRAAL
r = 0.001
rho = 1000
C = 0.04
K = (r^2)*pi*rho*C
g = 9.81
v0 = 13
x0 = 0
y0 = 2.5
hoek = 0
alpha = (hoek/180)*pi
x1 = v0*cos(alpha)
y1 = v0*sin(alpha)

%de tijdframe waarin we de baan van de waterstraal willen bekijken
tspan = [0 0.8];
%de vector van onze beginvoorwaarden
A = [x0,y0,x1,y1]
%de functie van differentialen met variabelen t (tijd) en N (matrix van
%posities)
f = @(t,N) [N(3); N(4);-K*N(3)*norm([N(3),N(4)]); 
                       -g-K*N(4)*norm([N(3),N(4)])];
%ode45 lost de differentiaalvergelijkingen op met de functies, domein tspan
%en beginvoorwaarden als input
[t, N] = ode45(f, tspan, A);

%plot alle punten horend bij domein tspan, opgelost door ode45

plot(N(:,1),N(:,2))