%%%%%%


clear all; close all; clc;

%% Figura 4.1. Representación red de Sevilla

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
n = 24;

long = [-5.99652514, -5.98410857, -5.98975246, -5.9400862 , -6.02587338,-5.89719261, -6.00104025, -6.00216903, -5.99878269, -6.00555537,...
    -5.92879841, -6.01571438, -6.03377484, -5.97959346, -5.96040422, -5.97620712, -6.02022949, -5.98410857, -6.00310964, -5.97191776,...
    -5.98884944, -5.91807502, -5.96068641, -6.0303885 ];
lat = [37.3525353 , 37.36021721, 37.3790728 , 37.35323365, 37.39234155, 37.42446589, 37.41119714, 37.38116786, 37.38326293, 37.38326293, ...
    37.40700701, 37.41399056, 37.39373826, 37.39443661, 37.38745306, 37.38815141, 37.3930399 , 37.37767609, ...
    37.40107099, 37.41790135,  37.38926878, 37.40962584, 37.36615323, 37.3790728 ];

lat(9) = 37.3820073;
long(9) = -5.9941967;
long(21) = -5.9959814;
long(17) = long(17) + 0.004;
lat(7) = lat(7) - 0.002;

scaler = 0.7;
I = imread('../red_sevilla/sevilla.png', 'BackgroundColor', [1 1 1]);
%colormap gray;
I = imresize(I, scaler);
[filas, columnas, ~] = size(I);
figure('Position', [100, 100, columnas, filas]);  
g = graph(zeros(n));
h = plot(g,'XData',scaler.*(long-mean(long)),'YData',scaler.*(lat-mean(lat)),'MarkerSize',5,'NodeFontSize',10,'Interpreter','latex');
hold on
h = image(xlim+scaler.*0.008,-0.8*ylim-0.001.*scaler,I); 
uistack(h,'bottom');
set(h,'AlphaData',0.8);
xticks([]); yticks([]); 
% Obtener dimensiones de la imagen
%
% Configurar los ejes para que coincidan con las dimensiones de la imagen
%axis([0.5, columnas+0.5, 0.5, filas+0.5]);
axis off; % Desactivar los ejes
figurename = sprintf('./figures/sevilla_nodos.png');
saveas(gcf, figurename);

%% Load results

clear all;

%betas = [0.01,0.1,0.2,0.22,0.24,0.26,0.28,0.3,0.35,0.4,0.41,0.43,0.45,0.5,0.6,0.7,0.75,0.8,0.9,0.95,1];
%with poverty
betas = [0.01,0.1,0.15,0.2,0.22,0.23,0.25,0.26,0.27,0.3,0.32,0.35,0.4,0.425,0.45,0.5,0.55,0.57,0.6,0.65,0.7];
betas = [0.15];
%betas = [0.1];
%poverty:
%lam 5: 0.57
%lam 7: 0.425
%lam 10: 0.32
%lam 12: 0.26,0.27
%lam 15: 0.22,0.23

%original:
betas = [0.01,0.1,0.105,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.2,0.21,0.22,0.23,0.25,0.26,0.27,0.28,0.3,0.32,0.35,0.4,0.425,0.45,0.5,0.55,0.57,0.6,0.65,0.7];
 %betas = [0.13];
%lam 5: 0.27-0.3
%lam 7: 0.2-0.22
%lam 10: 0.15- 0.16
%lam 12: 0.14
%lam 15: 0.11, 0.105

% betas = [0.01,0.1,0.12,0.125,0.13,0.132,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.22,0.23,0.24,0.26,0.28,0.3,0.32,0.33,0.35,0.4,0.41,0.43,0.45,0.5,0.6,0.7,0.8,0.9,1];
%
%betas = [0.16];
%crecimiento:
%lam 5: 
%lam 7: 
%lam 10: 
%lam 12: 
%lam 15: 0.132
%betas = [0.132];


lam = 10;
close all;

%betas = [0.15];

clc;
for bb=1:length(betas)
    beta_or = betas(bb);
    [n,link_cost,station_cost,link_capacity_slope,...
        station_capacity_slope,demand,prices,...
        op_link_cost,congestion_coef_stations,...
        congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,sigma,...
        a_max,candidates] = parameters_sevilla_network();
  %  filename = sprintf('../red_sevilla/sevilla_results/prev_symmetric_demandnomod_poverty_sol_beta=%d_lam=%d.mat',beta_or,lam);
    filename = sprintf('../red_sevilla/sevilla_results/prev_symmetric_demandnomod_sol_beta=%d_lam=%d.mat',beta_or,lam);
  %  filename = sprintf('../red_sevilla/sevilla_results/prev_symmetric_demandnomod_crecimiento_sol_beta=%d_lam=%d.mat',beta_or,lam);
    load(filename);
    att_d(bb) = 100*sum(sum(f.*demand))/sum(sum(demand));
    bud(bb) = budget;
    nroutes = 0;
    dis_rut = 0;
    for o=1:n
        for d=1:n
            dis(o,d) = sum(sum(travel_time.*squeeze(fij(:,:,o,d))))*demand(o,d);
            if f(o,d) > 0.01
                dis_rut = dis_rut + sum(sum(travel_time(squeeze(fij(:,:,o,d)) > 0.02)));
                nroutes = nroutes + 1;
            end
            uu(o,d) = demand(o,d)*alt_time(o,d)*fext(o,d);
        end
    end
    long_mean(bb) = dis_rut/nroutes;
    d_med(bb) = sum(sum(dis))/(sum(sum(f.*demand)));
    u_med(bb) = sum(sum(uu))/(sum(sum(fext.*demand)));
    disp(['gamma = ',num2str(1/beta_or)]);
    disp(['D.A. = ',num2str(100*sum(sum(f.*demand))/sum(sum(demand))),' %']);
    att_dem = round(100*sum(sum(f.*demand))/sum(sum(demand)),1);
   % disp(['att_dem = ',num2str(sum(sum(f.*demand))),' pax']);
    disp(['Pres. = ',num2str(budget)]);
    disp(['Arcos = ',num2str(sum(sum(a_prim > 0.1)))]);
    disp(['Nodos = ',num2str(sum(s_prim > 0.1))]);
    disp(['Cap. = ',num2str(sum(sum(a_prim)))]);
    disp(['distancia por pasajero red nueva = ',num2str(d_med(bb))]);
    disp(['distancia por pasajero red actual = ',num2str(u_med(bb))]);
    disp(['distancia por ruta = ',num2str(long_mean(bb))]);
    disp(['tiempo computacional = ',num2str(comp_time)]);
    disp('\n');
    
end


%% Figura 4.2. Representación red de Sevilla

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all;
figure('Position', [100, 100, 450, 300]);

subplot(211);
plot(bud,att_d,'-o','LineWidth',1.5); grid on; yl = ylabel('Demanda atraida [\%]','Interpreter','latex');
xl = xlabel('Presupuesto [EUR]','interpreter','latex');
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');

subplot(212);
plot(1./betas,att_d,'-o','LineWidth',1.5); grid on; xl = xlabel('$\gamma$','Interpreter','latex');
yl = ylabel('Demanda atraida [\%]','Interpreter','latex');
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');
saveas(gcf, './figures/sevilla_dem_pres_lam10_original.png'); % Guardar la figura en formato PNG


%% Figure 4.3

close all;
figure('Position', [100, 100, 450, 300]);

subplot(221);
plot(1./betas,d_med,'-o','LineWidth',1.5); 
hold on
plot(1./betas,u_med,'-o','LineWidth',1.5); grid on; 
xl = xlabel('$\gamma$','Interpreter','latex','FontSize',9); 
yl = ylabel('$\bar{d}_{PAX}$','Interpreter','latex','FontSize',9);
legend('Red nueva','Red competidora','Interpreter','latex','Location','best','FontSize',9);
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');


subplot(223);
plot(1./betas,d_med,'-o','LineWidth',1.5); 
hold on
plot(1./betas,u_med,'-o','LineWidth',1.5); grid on; 
xl = xlabel('$\gamma$','Interpreter','latex','FontSize',9); 
yl = ylabel('$\bar{d}_{PAX}$','Interpreter','latex','FontSize',9);
xlim([4 8.5]);
legend('Red nueva','Red competidora','Interpreter','latex','Location','best','FontSize',9 );
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');


subplot(222);
plot(1./betas,long_mean,'-o','LineWidth',1.5); grid on; xl = xlabel('$\gamma$','Interpreter','latex','FontSize',9); 
yl = ylabel('$\bar{d}_{ruta}$','Interpreter','latex','FontSize',9);

subplot(224);
plot(1./betas,long_mean,'-o','LineWidth',1.5); grid on; 
xl = xlabel('$\gamma$','Interpreter','latex','FontSize',9);
yl = ylabel('$\bar{d}_{ruta}$','Interpreter','latex','FontSize',9);
xlim([4 8.5]);

set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');

saveas(gcf, './figures/sevilla_mean_timeroute_lam10_original.png'); % Guardar la figura en formato PNG



%% Figura 4.4 Barrido en gamma

close all;
n = 24;

long = [-5.99652514, -5.98410857, -5.98975246, -5.9400862 , -6.02587338,-5.89719261, -6.00104025, -6.00216903, -5.99878269, -6.00555537,...
    -5.92879841, -6.01571438, -6.03377484, -5.97959346, -5.96040422, -5.97620712, -6.02022949, -5.98410857, -6.00310964, -5.97191776,...
    -5.98884944, -5.91807502, -5.96068641, -6.0303885 ];
lat = [37.3525353 , 37.36021721, 37.3790728 , 37.35323365, 37.39234155, 37.42446589, 37.41119714, 37.38116786, 37.38326293, 37.38326293, ...
    37.40700701, 37.41399056, 37.39373826, 37.39443661, 37.38745306, 37.38815141, 37.3930399 , 37.37767609, ...
    37.40107099, 37.41790135,  37.38926878, 37.40962584, 37.36615323, 37.3790728 ];

lat(9) = 37.3820073;
long(9) = -5.9941967;
long(21) = -5.9959814;
long(17) = long(17) + 0.004;
lat(7) = lat(7) - 0.002;

scaler = 0.7;
I = imread('../red_sevilla/sevilla.png', 'BackgroundColor', [1 1 1]);
I = imresize(I, scaler);
[filas, columnas, ~] = size(I);
figure('Position', [100, 100, 0.8.*columnas, 2.5.*filas]);  

lam = 10;
betas = [0.16,0.15,0.14];
for bb=1:length(betas)

    beta_or = betas(bb);
    filename = sprintf('../red_sevilla/sevilla_results/prev_symmetric_demandnomod_sol_beta=%d_lam=%d.mat',beta_or,lam);
    load(filename);
    
    a_b = zeros(n);
    a_b(a>1) = 1;
    a_b = a_b + a_b';
    a_b (a_b < 1.9) = 0;
    g = graph(a_prim);
    colormap parula
    colors = colormap;
    est_size = 3*ones(1,n);
    for i=1:n
        if s_prim(i) > 0
            est_size(i) = 4.*(s_prim(i).^0.2);
        end
    end
    colores_edg = colors( round(1e-2 + g.Edges.Weight .* length(colors)/max(g.Edges.Weight)) ,:);
    subplot(3,1,bb);
    h = plot(g,'XData',scaler.*(long-mean(long)),'YData',scaler.*(lat-mean(lat)),'MarkerSize',est_size, ...
        'LineWidth',0.7.*(g.Edges.Weight).^0.7,'EdgeColor',colores_edg,'EdgeAlpha',1, ...
        'Interpreter','latex');
    
    hold on
    h = image(xlim+scaler.*0.008,-0.8*ylim-0.001.*scaler,I); 
    uistack(h,'bottom');
    set(h,'AlphaData',0.8);
    xticks([]); yticks([]); 
    axis off; % Desactivar los ejes
    colorbar('TickLabelInterpreter','latex');
    set(gca, 'FontSize', 9);
    set(gca, 'TickLabelInterpreter', 'latex');

    [n,link_cost,station_cost,link_capacity_slope,...
        station_capacity_slope,demand,prices,...
        op_link_cost,congestion_coef_stations,...
        congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,sigma,...
        a_max,candidates] = parameters_sevilla_network();
    gam = 1./beta_or;
    att_d = 100*sum(sum(f.*demand))/sum(sum(demand));
    tit = ['$\gamma = ',num2str(round(gam,1)),'$, $\lambda = 10$, D.A. $ = ',num2str(round(att_d,1)),' \% $'];
    title(tit,'Interpreter','latex');

end




figurename = sprintf('./figures/topologia_original_lam=10.png');
saveas(gcf, figurename);


%% Figura 4.5. Barrido en lambda, caso original

close all;
n = 24;

long = [-5.99652514, -5.98410857, -5.98975246, -5.9400862 , -6.02587338,-5.89719261, -6.00104025, -6.00216903, -5.99878269, -6.00555537,...
    -5.92879841, -6.01571438, -6.03377484, -5.97959346, -5.96040422, -5.97620712, -6.02022949, -5.98410857, -6.00310964, -5.97191776,...
    -5.98884944, -5.91807502, -5.96068641, -6.0303885 ];
lat = [37.3525353 , 37.36021721, 37.3790728 , 37.35323365, 37.39234155, 37.42446589, 37.41119714, 37.38116786, 37.38326293, 37.38326293, ...
    37.40700701, 37.41399056, 37.39373826, 37.39443661, 37.38745306, 37.38815141, 37.3930399 , 37.37767609, ...
    37.40107099, 37.41790135,  37.38926878, 37.40962584, 37.36615323, 37.3790728 ];

lat(9) = 37.3820073;
long(9) = -5.9941967;
long(21) = -5.9959814;
long(17) = long(17) + 0.004;
lat(7) = lat(7) - 0.002;

scaler = 0.7;
I = imread('../red_sevilla/sevilla.png', 'BackgroundColor', [1 1 1]);
I = imresize(I, scaler);
[filas, columnas, ~] = size(I);
figure('Position', [100, 100, 0.8.*columnas, 2.5.*filas]);  

for ii=1:3
    switch ii
        case 1
            lam = 5;
            beta_or = 0.27;
        case 2
            lam = 10;
            beta_or = 0.15;
        case 3
            lam = 12;
            beta_or = 0.13;
    end
    filename = sprintf('../red_sevilla/sevilla_results/prev_symmetric_demandnomod_sol_beta=%d_lam=%d.mat',beta_or,lam);
    load(filename);
    
    a_b = zeros(n);
    a_b(a>1) = 1;
    a_b = a_b + a_b';
    a_b (a_b < 1.9) = 0;
    g = graph(a_prim);
    colormap parula
    colors = colormap;
    est_size = 3*ones(1,n);
    for i=1:n
        if s_prim(i) > 0
            est_size(i) = 4.*(s_prim(i).^0.2);
        end
    end
    colores_edg = colors( round(1e-2 + g.Edges.Weight .* length(colors)/max(g.Edges.Weight)) ,:);
    subplot(3,1,ii);
    h = plot(g,'XData',scaler.*(long-mean(long)),'YData',scaler.*(lat-mean(lat)),'MarkerSize',est_size, ...
        'LineWidth',0.7.*(g.Edges.Weight).^0.7,'EdgeColor',colores_edg,'EdgeAlpha',1, ...
        'Interpreter','latex');
    
    hold on
    h = image(xlim+scaler.*0.008,-0.8*ylim-0.001.*scaler,I); 
    uistack(h,'bottom');
    set(h,'AlphaData',0.8);
    xticks([]); yticks([]); 
    axis off; % Desactivar los ejes
    colorbar('TickLabelInterpreter','latex');
    set(gca, 'FontSize', 9);
    set(gca, 'TickLabelInterpreter', 'latex');
    
    [n,link_cost,station_cost,link_capacity_slope,...
        station_capacity_slope,demand,prices,...
        op_link_cost,congestion_coef_stations,...
        congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,sigma,...
        a_max,candidates] = parameters_sevilla_network();
    gam = 1./beta_or;
    att_d = 100*sum(sum(f.*demand))/sum(sum(demand));
    tit = ['$\gamma = ',num2str(round(gam,1)),'$, $\lambda = ', num2str(lam),'$, D.A. $ = ',num2str(round(att_d,1)),' \% $'];
    title(tit,'Interpreter','latex');
end



figurename = sprintf('./figures/topologia_original_barrido_lam.png');
saveas(gcf, figurename);









%% Figura 4.6. Representacion zonas deprimidas

close all;
n = 24;

long = [-5.99652514, -5.98410857, -5.98975246, -5.9400862 , -6.02587338,-5.89719261, -6.00104025, -6.00216903, -5.99878269, -6.00555537,...
    -5.92879841, -6.01571438, -6.03377484, -5.97959346, -5.96040422, -5.97620712, -6.02022949, -5.98410857, -6.00310964, -5.97191776,...
    -5.98884944, -5.91807502, -5.96068641, -6.0303885 ];
lat = [37.3525353 , 37.36021721, 37.3790728 , 37.35323365, 37.39234155, 37.42446589, 37.41119714, 37.38116786, 37.38326293, 37.38326293, ...
    37.40700701, 37.41399056, 37.39373826, 37.39443661, 37.38745306, 37.38815141, 37.3930399 , 37.37767609, ...
    37.40107099, 37.41790135,  37.38926878, 37.40962584, 37.36615323, 37.3790728 ];

lat(9) = 37.3820073;
long(9) = -5.9941967;
long(21) = -5.9959814;
long(17) = long(17) + 0.004;
lat(7) = lat(7) - 0.002;

scaler = 0.7;
I = imread('../red_sevilla/sevilla.png', 'BackgroundColor', [1 1 1]);
I = imresize(I, scaler);
[filas, columnas, ~] = size(I);
figure('Position', [100, 100, columnas, filas]);  

g = graph(zeros(n));
colormap parula
colors = colormap;

pond_coefs = [1.54535294, 1.54535294, 1.11218247, 1.72777031, 2.75490793, ...
        2.14294615, 1.41994992, 1, 1.11218247, 1.41994992,...
           2.14294615, 2.35701223, 2.75490793, 2.21242079, 3.        ,...
           1.59083705, 1.41994992, 1.01231062, 1.41994992, 2.35701223,...
           1.11218247, 2.14294615, 1.72777031, 1.45204153]';

h = plot(g,'XData',scaler.*(long-mean(long)),'YData',scaler.*(lat-mean(lat)),'MarkerSize',4.*(pond_coefs.^0.7), ...
    'Interpreter','latex');

hold on
h = image(xlim+scaler.*0.008,-0.8*ylim-0.001.*scaler,I); 
uistack(h,'bottom');
set(h,'AlphaData',0.8);
xticks([]); yticks([]); 
axis off; % Desactivar los ejes
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');


figurename = sprintf('./figures/Sevilla_renta_coefs.png');
saveas(gcf, figurename);




%% Figura 4.7. Rentas bajas, soluciones

close all;
n = 24;

long = [-5.99652514, -5.98410857, -5.98975246, -5.9400862 , -6.02587338,-5.89719261, -6.00104025, -6.00216903, -5.99878269, -6.00555537,...
    -5.92879841, -6.01571438, -6.03377484, -5.97959346, -5.96040422, -5.97620712, -6.02022949, -5.98410857, -6.00310964, -5.97191776,...
    -5.98884944, -5.91807502, -5.96068641, -6.0303885 ];
lat = [37.3525353 , 37.36021721, 37.3790728 , 37.35323365, 37.39234155, 37.42446589, 37.41119714, 37.38116786, 37.38326293, 37.38326293, ...
    37.40700701, 37.41399056, 37.39373826, 37.39443661, 37.38745306, 37.38815141, 37.3930399 , 37.37767609, ...
    37.40107099, 37.41790135,  37.38926878, 37.40962584, 37.36615323, 37.3790728 ];

lat(9) = 37.3820073;
long(9) = -5.9941967;
long(21) = -5.9959814;
long(17) = long(17) + 0.004;
lat(7) = lat(7) - 0.002;

scaler = 0.7;
I = imread('../red_sevilla/sevilla.png', 'BackgroundColor', [1 1 1]);
I = imresize(I, scaler);
[filas, columnas, ~] = size(I);
figure('Position', [100, 100, 0.8.*columnas, 2.5.*filas]);  

lam = 10;
for ii=1:2
    switch ii
        case 1
            beta_or = 0.32;
        case 2
            beta_or = 0.3;
    end
    filename = sprintf('../red_sevilla/sevilla_results/prev_symmetric_demandnomod_poverty_sol_beta=%d_lam=%d.mat',beta_or,lam);
    load(filename);
    
    a_b = zeros(n);
    a_b(a>1) = 1;
    a_b = a_b + a_b';
    a_b (a_b < 1.9) = 0;
    g = graph(a_prim);
    colormap parula
    colors = colormap;
    est_size = 3*ones(1,n);
    for i=1:n
        if s_prim(i) > 0
            est_size(i) = 4.*(s_prim(i).^0.2);
        end
    end
    colores_edg = colors( round(1e-2 + g.Edges.Weight .* length(colors)/max(g.Edges.Weight)) ,:);
    subplot(3,1,ii);
    h = plot(g,'XData',scaler.*(long-mean(long)),'YData',scaler.*(lat-mean(lat)),'MarkerSize',est_size, ...
        'LineWidth',0.7.*(g.Edges.Weight).^0.7,'EdgeColor',colores_edg,'EdgeAlpha',1, ...
        'Interpreter','latex');
    
    hold on
    h = image(xlim+scaler.*0.008,-0.8*ylim-0.001.*scaler,I); 
    uistack(h,'bottom');
    set(h,'AlphaData',0.8);
    xticks([]); yticks([]); 
    axis off; % Desactivar los ejes
    colorbar('TickLabelInterpreter','latex');
    set(gca, 'FontSize', 9);
    set(gca, 'TickLabelInterpreter', 'latex');
    
    [n,link_cost,station_cost,link_capacity_slope,...
        station_capacity_slope,demand,prices,...
        op_link_cost,congestion_coef_stations,...
        congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,sigma,...
        a_max,candidates] = parameters_sevilla_network();
    gam = 1./beta_or;
    att_d = 100*sum(sum(f.*demand))/sum(sum(demand));
    tit = ['$\gamma = ',num2str(round(gam,1)),'$, $\lambda = ', num2str(lam),'$, D.A. $ = ',num2str(round(att_d,1)),' \% $'];
    title(tit,'Interpreter','latex');
end


 figurename = sprintf('./figures/topologia_pobreza_lam=10.png');
 saveas(gcf, figurename);


%% Figura 4.8. Representación demográfica

 close all;
n = 24;

long = [-5.99652514, -5.98410857, -5.98975246, -5.9400862 , -6.02587338,-5.89719261, -6.00104025, -6.00216903, -5.99878269, -6.00555537,...
    -5.92879841, -6.01571438, -6.03377484, -5.97959346, -5.96040422, -5.97620712, -6.02022949, -5.98410857, -6.00310964, -5.97191776,...
    -5.98884944, -5.91807502, -5.96068641, -6.0303885 ];
lat = [37.3525353 , 37.36021721, 37.3790728 , 37.35323365, 37.39234155, 37.42446589, 37.41119714, 37.38116786, 37.38326293, 37.38326293, ...
    37.40700701, 37.41399056, 37.39373826, 37.39443661, 37.38745306, 37.38815141, 37.3930399 , 37.37767609, ...
    37.40107099, 37.41790135,  37.38926878, 37.40962584, 37.36615323, 37.3790728 ];

lat(9) = 37.3820073;
long(9) = -5.9941967;
long(21) = -5.9959814;
long(17) = long(17) + 0.004;
lat(7) = lat(7) - 0.002;

scaler = 0.7;
I = imread('../red_sevilla/sevilla.png', 'BackgroundColor', [1 1 1]);
I = imresize(I, scaler);
[filas, columnas, ~] = size(I);
figure('Position', [100, 100, columnas, filas]);  

g = graph(zeros(n));
colormap parula
colors = colormap;

crec_coefs = [1.6, 1.6, 1.1469802107427398, 0.9, 1.2081587290019313, 0.9661783579052806, 1.0802156586966714, ...
    0.9, 1.1469802107427398, 1.0802156586966714, 0.9661783579052806, 0.9989307690507944,...
    1.2081587290019313, 0.9, 1.0730507219686063, 0.9, 1.0802156586966714, 1.3414585127763425, ...
    1.0802156586966714, 0.9989307690507944, 1.1469802107427398, 0.9661783579052806, 0.9, 1.1224800389355853]';

h = plot(g,'XData',scaler.*(long-mean(long)),'YData',scaler.*(lat-mean(lat)),'MarkerSize',5.*(crec_coefs.^1.1), ...
    'Interpreter','latex');

hold on
h = image(xlim+scaler.*0.008,-0.8*ylim-0.001.*scaler,I); 
uistack(h,'bottom');
set(h,'AlphaData',0.8);
xticks([]); yticks([]); 
axis off; % Desactivar los ejes
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');


figurename = sprintf('./figures/Sevilla_crecimiento_coefs.png');
saveas(gcf, figurename);


%% Figura 4.9. Demografía, soluciones

close all;
n = 24;

long = [-5.99652514, -5.98410857, -5.98975246, -5.9400862 , -6.02587338,-5.89719261, -6.00104025, -6.00216903, -5.99878269, -6.00555537,...
    -5.92879841, -6.01571438, -6.03377484, -5.97959346, -5.96040422, -5.97620712, -6.02022949, -5.98410857, -6.00310964, -5.97191776,...
    -5.98884944, -5.91807502, -5.96068641, -6.0303885 ];
lat = [37.3525353 , 37.36021721, 37.3790728 , 37.35323365, 37.39234155, 37.42446589, 37.41119714, 37.38116786, 37.38326293, 37.38326293, ...
    37.40700701, 37.41399056, 37.39373826, 37.39443661, 37.38745306, 37.38815141, 37.3930399 , 37.37767609, ...
    37.40107099, 37.41790135,  37.38926878, 37.40962584, 37.36615323, 37.3790728 ];

lat(9) = 37.3820073;
long(9) = -5.9941967;
long(21) = -5.9959814;
long(17) = long(17) + 0.004;
lat(7) = lat(7) - 0.002;

scaler = 0.7;
I = imread('../red_sevilla/sevilla.png', 'BackgroundColor', [1 1 1]);
I = imresize(I, scaler);
[filas, columnas, ~] = size(I);
figure('Position', [100, 100, 0.8.*columnas, 2.5.*filas]);  

lam = 10;
for ii=1:2
    switch ii
        case 1
            beta_or = 0.17;
        case 2
            beta_or = 0.15;
    end
    filename = sprintf('../red_sevilla/sevilla_results/prev_symmetric_demandnomod_crecimiento_sol_beta=%d_lam=%d.mat',beta_or,lam);
    load(filename);
    
    a_b = zeros(n);
    a_b(a>1) = 1;
    a_b = a_b + a_b';
    a_b (a_b < 1.9) = 0;
    g = graph(a_prim);
    colormap parula
    colors = colormap;
    est_size = 3*ones(1,n);
    for i=1:n
        if s_prim(i) > 0
            est_size(i) = 4.*(s_prim(i).^0.2);
        end
    end
    colores_edg = colors( round(1e-2 + g.Edges.Weight .* length(colors)/max(g.Edges.Weight)) ,:);
    subplot(3,1,ii);
    h = plot(g,'XData',scaler.*(long-mean(long)),'YData',scaler.*(lat-mean(lat)),'MarkerSize',est_size, ...
        'LineWidth',0.7.*(g.Edges.Weight).^0.7,'EdgeColor',colores_edg,'EdgeAlpha',1, ...
        'Interpreter','latex');
    
    hold on
    h = image(xlim+scaler.*0.008,-0.8*ylim-0.001.*scaler,I); 
    uistack(h,'bottom');
    set(h,'AlphaData',0.8);
    xticks([]); yticks([]); 
    axis off; % Desactivar los ejes
    colorbar('TickLabelInterpreter','latex');
    set(gca, 'FontSize', 9);
    set(gca, 'TickLabelInterpreter', 'latex');
    
    [n,link_cost,station_cost,link_capacity_slope,...
        station_capacity_slope,demand,prices,...
        op_link_cost,congestion_coef_stations,...
        congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,sigma,...
        a_max,candidates] = parameters_sevilla_network();
    gam = 1./beta_or;
    att_d = 100*sum(sum(f.*demand))/sum(sum(demand));
    tit = ['$\gamma = ',num2str(round(gam,1)),'$, $\lambda = ', num2str(lam),'$, D.A. $ = ',num2str(round(att_d,1)),' \% $'];
    title(tit,'Interpreter','latex');
end


  figurename = sprintf('./figures/topologia_crecimiento_lam=10.png');
  saveas(gcf, figurename);



%% Funciones

function [n,link_cost,station_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,sigma,...
    a_max,candidates,pond_coefs_mat] = parameters_sevilla_network()

    n = 24;
    
    %candidates to construct a link for each neighbor
    candidates = {
    [4,23,2,18,3,9,8,10,24,5,17];
    [1,4,23,18,3,9,8,5,24];
    [1,2,18,16,14,21,19,9,8,24];
    [1,2,23,16,15,20,11,22,6];
    [1,2,10,9,21,14,17,12,13,24];
    [4,22,16,14,19,7,20];
    [12,13,17,19,21,14,11,22,6,20];
    [1,2,3,9,10,24];
    [1,2,3,16,21,20,19,17,5,10,8,12];
    [1,8,9,20,19,12,17,5,24,21];
    [4,22,20,7,19,14,16,15,23];
    [13,5,17,10,9,19,14,7,20];
    [24,5,17,19,7,12];
    [21,3,18,16,11,22,6,20,7,12,19,17,5,15];
    [18,23,4,22,11,20,14,16];
    [18,23,4,15,11,6,14,21,9,3,22];
    [5,24,1,10,9,21,14,19,7,12,13];
    [2,23,15,16,14,21,19,3,1];
    [13,17,10,9,3,18,21,14,11,6,20,7,12];
    [6,22,11,4,15,14,21,9,10,19,7,12];
    [3,18,23,16,14,20,7,19,17,5,10,9];
    [4,6,20,7,14,16,11,15,23];
    [1,4,22,11,15,16,21,18,24,2];
    [1,2,23,3,8,10,17,5,13];
    };
    

    population_file = readtable('../red_sevilla/population.xlsx');
    population = table2array(population_file(1:24,2));
    coordinates = readtable('../red_sevilla/coordenadas_Sevilla.xlsx');
    coor_x = table2array(coordinates(1:24,3));
    coor_y = table2array(coordinates(1:24,7));
    rng(1,"twister"); %seed
    distance = 1e6.*ones(n);
    for i=1:n
        distance(i,i) = 0;
        cand = candidates(i);
        cand = cand{1};
        cand = cand(cand > i);
        for j=i+1:n
            if sum(j == cand) > 0
                distance(i,j) = haversine(coor_y(i), coor_x(i), coor_y(j), coor_x(j));
                distance(j,i) = distance(i,j);
            end
            non_stop = rand < 0.4;

            alt_cost(i,j) = haversine(coor_y(i), coor_x(i), coor_y(j), coor_x(j));
            alt_cost(i,j) = alt_cost(i,j) + 0.2*non_stop*alt_cost(i,j);
            alt_cost(j,i) = alt_cost(i,j);
        end
    end
    demand = [0, 272, 272, 272, 272, 553, 272, 272, 272, 553, 553, 272, 553, 272, 553, 553, 553, 272, 937, 937, 937, 937, 937, 937;
           272, 0, 272, 272, 272, 553, 272, 272, 272, 553, 553, 272, 553, 272, 553, 553, 553, 272, 937, 937, 937, 937, 937, 937;
           327, 327, 0, 327, 327, 664, 327, 327, 327, 664, 664, 327, 664, 327, 664, 664, 664, 327, 1125, 1125, 1125, 1125, 1125, 1125;
           185, 185, 185, 0, 185, 376, 185, 185, 185, 376, 376, 185, 376, 185, 376, 376, 376, 185, 637, 637, 637, 637, 637, 637;
           272, 272, 272, 272, 0, 553, 272, 272, 272, 553, 553, 272, 553, 272, 553, 553, 553, 272, 937, 937, 937, 937, 937, 937;
           225, 225, 225, 225, 225, 0, 225, 225, 225, 188, 188, 225, 188, 225, 188, 188, 188, 225, 284, 284, 284, 284, 284, 284;
           283, 283, 283, 283, 283, 575, 0, 283, 283, 575, 575, 283, 575, 283, 575, 575, 575, 283, 975, 975, 975, 975, 975, 975;
           272, 272, 272, 272, 272, 553, 272, 0, 272, 553, 553, 272, 553, 272, 553, 553, 553, 272, 937, 937, 937, 937, 937, 937;
           272, 272, 272, 272, 272, 553, 272, 272, 0, 553, 553, 272, 553, 272, 553, 553, 553, 272, 937, 937, 937, 937, 937, 937;
           511, 511, 511, 511, 511, 428, 511, 511, 511, 0, 428, 511, 428, 511, 428, 428, 428, 511, 645, 645, 645, 645, 645, 645;
           225, 225, 225, 225, 225, 188, 225, 225, 225, 188, 0, 225, 188, 225, 188, 188, 188, 225, 284, 284, 284, 284, 284, 284;
           272, 272, 272, 272, 272, 553, 272, 272, 272, 553, 553, 0, 553, 272, 553, 553, 553, 272, 937, 937, 937, 937, 937, 937;
           306, 306, 306, 306, 306, 257, 306, 306, 306, 257, 306, 257, 0, 257, 306, 306, 306, 257, 387, 387, 387, 387, 387, 387;
           294, 294, 294, 294, 294, 597, 294, 294, 294, 597, 597, 294, 597, 0, 597, 597, 597, 297, 1012, 1012, 1012, 1012, 1012, 1012;
           409, 409, 409, 409, 409, 342, 409, 409, 409, 342, 342, 409, 342, 409, 0, 342, 342, 409, 516, 516, 516, 516, 516, 516;
           511, 511, 511, 511, 511, 428, 511, 511, 511, 428, 428, 511, 428, 511, 428, 0, 428, 511, 645, 645, 645, 645, 645, 645;
           429, 429, 429, 429, 429, 360, 429, 429, 429, 306, 360, 429, 360, 429, 360, 360, 0, 429, 542, 542, 542, 542, 542, 542;
           272, 272, 272, 272, 272, 553, 272, 272, 272, 553, 553, 272, 553, 272, 553, 553, 553, 0, 937, 937, 937, 937, 937, 937;
           675, 675, 675, 675, 675, 730, 675, 675, 675, 730, 730, 675, 730, 675, 730, 730, 730, 675, 660, 660, 660, 660, 660, 660;
           879, 879, 879, 879, 879, 952, 879, 879, 879, 952, 952, 879, 952, 879, 952, 952, 952, 879, 860, 0, 860, 860, 860, 860;
           715, 715, 715, 715, 715, 775, 715, 715, 715, 775, 775, 715, 775, 715, 775, 775, 775, 715, 700, 700, 700, 700, 700, 700;
           511, 511, 511, 511, 511, 553, 511, 511, 511, 553, 553, 511, 553, 511, 553, 553, 553, 511, 500, 500, 500, 0, 500, 500;
           675, 675, 675, 675, 675, 730, 675, 675, 675, 730, 730, 675, 730, 675, 730, 730, 730, 675, 660, 660, 660, 660, 0, 660;
           675, 675, 675, 675, 675, 730, 675, 675, 675, 730, 730, 675, 730, 675, 730, 730, 730, 675, 660, 660, 660, 660, 660, 0];

    pond_coefs = [1.54535294, 1.54535294, 1.11218247, 1.72777031, 2.75490793, ...
        2.14294615, 1.41994992, 1, 1.11218247, 1.41994992,...
           2.14294615, 2.35701223, 2.75490793, 2.21242079, 3.        ,...
           1.59083705, 1.41994992, 1.01231062, 1.41994992, 2.35701223,...
           1.11218247, 2.14294615, 1.72777031, 1.45204153]';
    
    pond_coefs_tens(1,:,:) = pond_coefs.*ones(1,n);
    pond_coefs_tens(2,:,:) = permute(pond_coefs_tens(1,:,:),[1 3 2]);
    pond_coefs_mat = squeeze(permute(max(pond_coefs_tens(1,:,:),pond_coefs_tens(2,:,:)),[2,3,1]));

    crec_coefs = [1.6, 1.6, 1.1469802107427398, 0.9, 1.2081587290019313, 0.9661783579052806, 1.0802156586966714, ...
        0.9, 1.1469802107427398, 1.0802156586966714, 0.9661783579052806, 0.9989307690507944,...
        1.2081587290019313, 0.9, 1.0730507219686063, 0.9, 1.0802156586966714, 1.3414585127763425, ...
        1.0802156586966714, 0.9989307690507944, 1.1469802107427398, 0.9661783579052806, 0.9, 1.1224800389355853];

    for o=1:n
        for d=1:n
            demand(o,d) = crec_coefs(o)*crec_coefs(d)*demand(o,d);
        end
    end

   % demand = demand.*pond_coefs_mat;
    %demand = max(demand,demand');
    
    %fixed cost for constructing links
    link_cost = 1e6.*distance./(365.25*25);
    
    %fixed cost for constructing stations
    station_cost = 1e3.*population./(365.25*25);
    
    link_capacity_slope = 0.3.*link_cost; 
    station_capacity_slope = 0.2.*station_cost;
    
    
    % Op Link Cost
    op_link_cost = 4.*distance;
    
    % Congestion Coefficients
    congestion_coef_stations = 0.1 .* ones(1, n);
    congestion_coef_links = 0.1 .* ones(n);
    
    % Prices
    prices = 0.1.*(distance).^(0.7);
    %prices = zeros(n);
    
    % Travel Time
    travel_time = 60 .* distance ./ 30; % Time in minutes
    
    % Alt Time
    alt_time = 60 .* alt_cost ./ 30; % Time in minutes
    alt_price = 0.1.*(alt_cost).^(0.7); %price
    
    
    a_nom = 588;             
    
    tau = 0.57;
    sigma = 0.25;
    a_max = 1e9;
    eps = 1e-3;
end

function distancia = haversine(lat1, lon1, lat2, lon2)
    % Convierte las coordenadas de grados a radianes
    lat1 = deg2rad(lat1);
    lon1 = deg2rad(lon1);
    lat2 = deg2rad(lat2);
    lon2 = deg2rad(lon2);

    % Diferencias en coordenadas
    dlat = lat2 - lat1;
    dlon = lon2 - lon1;

    % Fórmula haversine
    a = sin(dlat/2)^2 + cos(lat1) * cos(lat2) * sin(dlon/2)^2;
    c = 2 * atan2(sqrt(a), sqrt(1-a));

    % Radio de la Tierra en kilómetros (aproximado)
    radio_tierra = 6371;

    % Calcula la distancia
    distancia = radio_tierra * c;
end
