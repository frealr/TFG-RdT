

clear all; close all; clc;
% En cada sección de este fichero se representan las figuras presentes
% en el capítulo 2 de este TFG, ordenadas según aparecen en el documento.

% El convenio para las dimensiones de las figuras será el siguiente:
% - Tamaño de letra:
% El código necesario es el siguiente:


% - Formato de la ventana:
% El código necesario es el siguiente

% - Grosor de línea:
% Cuando el grosor de línea no represente una magnitud en la figura,
% el grosor por defecto será el siguiente: 

% El código necesario es el siguiente

% -Código de color:
% El código necesario es el siguiente

% -Formas de los puntos y tipos de línea:
% El código necesario es el siguiente

clear all; close all; clc;

%% Figura 3.1. Modelo de congestión relativo

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = 0:0.01:10;
t0 = 3;
a = 10;
eps = 1.5e-1;
t = t0 + 0.2*x./(a - x + eps);
figure('Position', [100, 100, 450, 300]);
plot(x,t,'LineWidth',1.5);
ylim([0 20]);
grid minor;
ylabel('$t(F_{ij}|A_{ij})$ [min]','Interpreter','latex');
xticks([0,0.25*a,0.5*a,0.75*a,a]);
xticklabels({'$F_{ij}=0$','$F_{ij}=0.25A_{ij}$','$F_{ij}=0.5A_{ij}$','$F_{ij}=0.75A_{ij}$','$F_{ij}=A_{ij}$'});
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');
saveas(gcf, 'congestion.png'); % Guardar la figura en formato PNG

%% Figura 3.2. Modelo de congestión absoluto

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
x = 0:0.01:10;
t0 = 3;
a = 10;
eps = 1.5e-1;
t = t0 + 2./(a - x + eps);
figure('Position', [100, 100, 450, 300]);
plot(x,t,'LineWidth',1.5);
ylim([0 20]);
grid minor;
ylabel('$t(F_{ij}|A_{ij})$ [min]','Interpreter','latex');
xticks([0,2.5,5,7.5,10]);
xticklabels({'$A_{ij}-F_{ij}=10$','$A_{ij}-F_{ij}=7.5$', ...
    '$A_{ij}-F_{ij}=5$','$A_{ij}-F_{ij}=2.5$','$A_{ij}-F_{ij}=0$'});
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');
saveas(gcf, 'congestion_absoluto.png'); % Guardar la figura en formato PNG

%% Figuras 3.3 y 3.4. Análisis de sensibilidad

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;


clc;
close all;
clear all;
cvx_begin quiet
cvx_end
betas= [1.25,1.5];
lams = [5,6];
list_of_parameters = {'demand','alt_time','prices','alt_price','fixed_costs',...
    'linear_costs','op_cost'};
param_name = string('op_cost');
close all;
sens_coefs = 0.8:0.05:1.2;
alfa = 0.6;
dm_pax = 1.2;
dm_op = 0.008;
figure('Position', [100, 100, 675, 450]);

%for pp = 1:length(list_of_parameters)
for pp = 5:7
    param_name = string(list_of_parameters{pp});
    subplot(2,2,pp-4);

    if (pp == 2) || (pp == 3) || (pp == 4) || (pp == 7)
        linetype = '-o';
    else
        linetype = '--o';
    end
    obj_gap = zeros(length(sens_coefs),length(lams),length(betas));
    cons_links = zeros(length(sens_coefs),length(lams),length(betas));
    abs_dem = zeros(length(sens_coefs),length(lams),length(betas));
    
    obj_noms = zeros(length(betas),length(lams));
    buds_nom = obj_noms;
    
    for bb=1:length(betas)
        for ll=1:length(lams)
            beta = betas(bb);
            lam = lams(ll);
            filename = sprintf('../red_9_nodos/uthopy_results/sol_beta=%d_lam=%d.mat',beta,lam);
            load(filename);
            obj_noms(bb,ll) = obj_val;
            bud_noms(bb,ll) = budget;
        end
    end
    
    gaps_less = zeros(length(sens_coefs),length(betas),length(lams));
    gaps_more = gaps_less;
    gaps_bud_more = gaps_less;
    gaps_bud_less = gaps_less;
    pond_coefs = gaps_less;
    
    for bb = 1:length(betas)
        beta = betas(bb);
        for ll = 1:length(lams)
            lam = lams(ll);
            if ~((beta==1.5) & (lam == 6))
                for ss=1:length(sens_coefs)
                    sens_coef = sens_coefs(ss);
                    [n,link_cost,station_cost,link_capacity_slope,...
                    station_capacity_slope,demand,prices,...
                    load_factor,op_link_cost,congestion_coef_stations,...
                    congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,eta,...
                    a_max,candidates] = parameters_9node_network();
                    if sens_coef == 1
                        filename = sprintf('../red_9_nodos/uthopy_results/sol_beta=%d_lam=%d_sens_analysis_%s_coef=%d_lessb.mat',beta,lam,param_name,sens_coef);
                        load(filename);
                        gaps_less(ss,bb,ll) = (obj_val - obj_noms(bb,ll))./obj_noms(bb,ll);
                        gaps_more(ss,bb,ll) = gaps_less(ss);
                        pond_coefs(ss,bb,ll) = 0.5;
        
                    else
                        
                        if param_name == 'alt_time'
                           % alt_time = sens_coef.*alt_time;
                        elseif param_name == 'demand'
                            demand = sens_coef.*demand;
                        elseif param_name == 'prices'
                           % prices = sens_coef.*prices;
                        elseif param_name == 'alt_price'
                           % alt_price = sens_coef.*alt_price;
                        elseif param_name == 'fixed_costs'
                            link_cost = sens_coef.*link_cost;
                            station_cost = sens_coef.*station_cost;
                        elseif param_name == 'linear_costs'
                            link_capacity_slope = sens_coef.*link_capacity_slope;
                            station_capacity_slope = sens_coef.*station_capacity_slope;
                        elseif param_name == 'op_cost'
                           % op_link_cost = sens_coef.*op_link_cost;
                        end
                        filename = sprintf('../red_9_nodos/uthopy_results/sol_beta=%d_lam=%d_sens_analysis_%s_coef=%d_lessb.mat',beta,lam,param_name,sens_coef);
                        load(filename);
                        [obj_val,pax_obj,op_obj] = get_obj_val(alfa, op_link_cost,...
                            congestion_coef_links, ...
                            congestion_coef_stations,travel_time,prices,alt_time,alt_price,a_prim,delta_a, ...
                            s_prim,delta_s,fij,f,fext,demand,dm_pax,dm_op);
                        gaps_less(ss,bb,ll) = (obj_val - obj_noms(bb,ll))./obj_noms(bb,ll);
                        b_less = budget;
                        gaps_bud_less(ss,bb,ll) = (budget-bud_noms(bb,ll))./bud_noms(bb,ll);
                        filename = sprintf('../red_9_nodos/uthopy_results/sol_beta=%d_lam=%d_sens_analysis_%s_coef=%d_moreb.mat',beta,lam,param_name,sens_coef);
                        load(filename);
                        [obj_val,pax_obj,op_obj] = get_obj_val(alfa, op_link_cost,...
                            congestion_coef_links, ...
                            congestion_coef_stations,travel_time,prices,alt_time,alt_price,a_prim,delta_a, ...
                            s_prim,delta_s,fij,f,fext,demand,dm_pax,dm_op);
                        gaps_bud_more(ss,bb,ll) = (budget-bud_noms(bb,ll))./bud_noms(bb,ll);
                        gaps_more(ss,bb,ll) = (obj_val - obj_noms(bb,ll))./obj_noms(bb,ll);
                        b_more = budget;
                        pond_coefs(ss,bb,ll) = (bud_noms(bb,ll)-b_more)./(b_less-b_more);
                     end
                end
            end
        end
    end
    colores = {'#0072BD','#D95319',	'#EDB120'};
    leg = {};
    
    for bb=1:length(betas)
        for ll=1:length(lams)
            beta = betas(bb);
            lam = lams(ll);
            if ~((beta==1.5) & (lam == 6))
    %              plot(sens_coefs,100.*gaps_less(:,bb,ll),'-o');
    %              hold on;
                corr_lower = 100.*gaps_less(:,bb,ll) - 0.20466.*100.*gaps_bud_less(:,bb,ll);
                corr_upper = 100.*gaps_more(:,bb,ll) + 0.158465.*100.*gaps_bud_more(:,bb,ll);
            
                %plot(corr_lower,'-o');
               % plot(corr_upper,'-o');
            if (beta == 1.25) && (lam ==5)
                color = colores{1};
            elseif (beta == 1.25) && (lam == 6)
                color = colores{2};
            elseif (beta == 1.5) && (lam == 5)
                color = colores{3};
            end
               %para alt_time, prices, alt_price, op_cost
                %plot(100.*(sens_coefs-1),pond_coefs(:,bb,ll).*corr_lower + (1-pond_coefs(:,bb,ll)).*corr_upper,'-o','LineWidth',1.5);
                %para demand, fixed_costs, linear_costs
                plot(100.*(sens_coefs(1:5)-1),pond_coefs(1:5,bb,ll).*corr_lower(1:5) + (1-pond_coefs(1:5,bb,ll)).*corr_upper(1:5),linetype,'LineWidth',1.5,'Color',color);
                hold on;
                plot(100.*(sens_coefs(5:end)-1),pond_coefs(5:end,bb,ll).*corr_lower(5:end) + (1-pond_coefs(5:end,bb,ll)).*corr_upper(5:end),'-o','LineWidth',1.5,'Color',color);
                
                
               
                 hold on;
                 
                if param_name == 'alt_time'
                   param_name2 = 't_{ext}^{od} $ $ \forall (o,d) \in \mathcal{K}';
                elseif param_name == 'demand'
                    param_name2 = 'w^{od} $ $ \forall (o,d) \in \mathcal{K}';
                elseif param_name == 'prices'
                   param_name2 = 'p_{ij} $ $ \forall (i,j) \in \mathcal{A}';
                elseif param_name == 'alt_price'
                   param_name2 = 'p_{ext}^{od} $ $ \forall (o,d) \in \mathcal{K}';
                elseif param_name == 'fixed_costs'
                    param_name2 = 'c_{S_i}$ $\forall i \in \mathcal{S}$, $c_{A_{ij}} $ $ \forall (i,j) \in \mathcal{A} ';
                elseif param_name == 'linear_costs'
                    param_name2 = '\bar{c}_{S_i}$ $\forall i \in \mathcal{S}$, $\bar{c}_{A_{ij}} $ $ \forall (i,j) \in \mathcal{A}';
                elseif param_name == 'op_cost'
                   param_name2 = 'c_{o_{ij}} $ $ \forall (i,j) \in \mathcal{A}';
                end
                 
                 %   plot(100.*(sens_coefs-1),100.*(pond_coefs(:,bb,ll).*gaps_less(:,bb,ll) + (1-pond_coefs(:,bb,ll)).*gaps_more(:,bb,ll)),'-o');
              %  hold on;
                leg_el = [''];
                leg = [leg,leg_el];
                leg_el = ['\lambda = ',num2str(lam),', \beta = ',num2str(beta)];
                leg = [leg,leg_el];
                %legend('more budget sol','less budget sol','same budget interpolation');
                
                
            end
        end
    end
    
    
    tit = sprintf('%d. Sensibilidad hacia $ %s $',pp-4,param_name2);
    tit = title(tit,'interpreter','latex');
  %  tit.FontSize = 14;
    xl = sprintf('Cambio en $ %s $ [%%] ',param_name2);
    xl = strrep(xl, '%', '\%');
   % xl = [xl,' [\%%]'];
    xl = xlabel(xl,'Interpreter','latex');
 %   xl.FontSize = 14;
    yl = ylabel('Cambio en el objetivo [\%]','Interpreter','latex');
 %   yl.FontSize = 12;
    grid on;
  %  lg = legend(leg,'Location','northwest');
    lg = legend('','$\lambda = 5$, $\beta = 1.25$','','$\lambda = 6$, $\beta = 1.25$','$\lambda = 5$, $\beta = 1.5$','Interpreter','latex','Location','best');
    lg.FontSize = 9;

    set(gca, 'FontSize', 9);
    set(gca, 'TickLabelInterpreter', 'latex');


end

saveas(gcf, 'sensibilidad_2.png'); % Guardar la figura en formato PNG



%% Figura 3.5. Diagrama de pareto
close all;
alfas = 0.1:0.1:0.9;
lam = 6;
nom_bud = 70000;
alfa = 1;
filename = sprintf('../red_9_nodos/uthopy_results/sol_MIP_budget=%s_lam=%d_uthopy_alfa=%d.mat',sprintf('%d',nom_bud),lam,alfa);
load(filename);

best_pax = pax_obj;
worst_op = op_obj;

alfa = 0;
filename = sprintf('../red_9_nodos/uthopy_results/sol_MIP_budget=%d_lam=%d_uthopy_alfa=%d.mat',nom_bud,lam,alfa);
load(filename);

best_op = op_obj;
worst_pax = pax_obj;

dm_pax = worst_pax - best_pax;
dm_op = worst_op - best_op;

clc;
alfas = 0.1:0.1:0.9;
alfas = [0,0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
lam = 6;
nom_bud = 70000;
close all;
obs_pax = zeros(length(alfas),1);
obs_op = obs_pax;


for aa=1:length(alfas)
    alfa = alfas(aa);
    filename = sprintf('../red_9_nodos/uthopy_results/sol_MIP_budget=%d_lam=%d_uthopy_alfa=%d.mat',nom_bud,lam,alfa);
    load(filename);
    rel_pax = (pax_obj - best_pax)/(dm_pax);
    rel_op = (op_obj - best_op)/(dm_op);
    ut_distance = sqrt(rel_pax.^2 + rel_op.^2);
    obs_pax(aa) = rel_pax;
    obs_op(aa) = rel_op;
    disp(['alfa = ',num2str(alfa),', euclidean distance to uthopical point = ',num2str(ut_distance),', nlinks = ',num2str(sum(sum((a > 1))))]);
end
figure('Position', [100, 100, 450, 300]);

plot(obs_pax,obs_op,'-o','LineWidth',1.5);
xlim([0 1]); ylim([0 1]);
grid on; xlabel('$PAX$ (normalizado)','interpreter','latex'); ylabel('$OP$ (normalizado)','interpreter','latex');
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');
saveas(gcf, 'pareto_utopia.png'); % Guardar la figura en formato PNG

%% Figura 3.6. Comparativa


betas = [1,3,5,7,10,12];
lams = [3,5];

betas = [1,3,5,7,10,12];
lams = [3,5];
total_demand = sum(sum(demand));

betas = [0.1,0.25,0.5,0.75,1,1.25,2];
betas = [0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5];
lams = [5,6];



 
budgets = zeros(length(betas),length(lams));
budgets_MIP = budgets;
att_dem = zeros(length(betas),length(lams));
att_dem_MIP = att_dem;
nlinks = zeros(length(betas),length(lams));
nlinks_MIP = nlinks;
obj_val_relax = zeros(length(betas),length(lams));
obj_val_MIP = obj_val_relax;
time = zeros(length(betas),length(lams));
time_MIP = time;
close all;
n = 9;
figure('Position', [100, 100, 675, 450]);
for i = 1:length(betas)
    beta = betas(i);
    for j = 1:length(lams)
        lam = lams(j);
        %filename = sprintf('./results/betas/sol_beta=%d_lam=%d.mat',beta,lam);
        filename = sprintf('../red_9_nodos/uthopy_results/sol_beta=%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIP_sparsityaprim_budget=beta%d_lam=%d.mat',beta,lam);
         %filename = sprintf('./results/betas/sol_log2_beta_sparsityprim=%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIPmod_budget=beta%d_lam=%d.mat',beta,lam);
        load(filename);
        disp(['beta = ',num2str(beta),' lam = ',num2str(lam)]);
        disp(['objs: pax = ',num2str(pax_obj),', op = ',num2str(op_obj),' obj = ', num2str(obj_val)]);
        budgets(i,j) = budget;
        att_dem(i,j) = sum(sum(f.*demand));
        eps = 1e-3;
        nlinks(i,j) = sum(sum(a_prim > eps));
        obj_val_relax(i,j) = obj_val;
        time(i,j) = comp_time;

        %filename = sprintf('./results/betas/sol_beta_sparsityprim=%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_log_beta_sparsityprim=%d_lam=%d.mat',beta,lam);
        filename = sprintf('../red_9_nodos/uthopy_results/sol_MIP_beta=%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIPnoLuis_sparsityaprim_budget=beta%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIPmod2_budget=beta%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIP_budget=beta%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIP2iter_budget=beta%d_lam=%d.mat',beta,lam);
        load(filename);
        budgets_MIP(i,j) = budget;
        att_dem_MIP(i,j) = sum(sum(f.*demand));
        %nlinks_MIP(i,j) = sum(sum(a_bin));
        eps = 1e-3;
        nlinks_MIP(i,j) = sum(sum(a_prim > eps));
        obj_val_MIP(i,j) = obj_val;
        time_MIP(i,j) = comp_time;

    end
end




subplot(221);
color = {'blue','red'};
for i = 1:length(lams)
    c = color(i);
    plot(betas,nlinks(:,i),'-o','LineWidth',1.5,'Color',string(c));
    hold on;
    plot(betas,nlinks_MIP(:,i),'--o','LineWidth',1.5,'Color',string(c));
    hold on;
end

xl = xlabel('$\beta$','interpreter','latex'); yl = ylabel('arcos construidos','interpreter','latex'); grid on;
tit = title('1. Arcos construidos vs $\beta$ para diferentes $\lambda$','interpreter','latex');
lg = legend('$\lambda = 5$','$\lambda = 5$ MIP','$\lambda = 6$','$\lambda = 6$ MIP','Location','best','interpreter','latex');


tit.FontSize = 9;
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');
lg.FontSize = 9;

subplot(222);
color = {'blue','red'};
for i = 1:length(lams)
    c = color(i);
    plot(betas,100.*att_dem(:,i)./total_demand,'-o','LineWidth',1.5,'Color',string(c));
    hold on;
    plot(betas,100.*att_dem_MIP(:,i)./total_demand,'--o','LineWidth',1.5,'Color',string(c));
    hold on;
end

xl = xlabel('$\beta$','interpreter','latex'); yl = ylabel('demanda absorbida [\%]','interpreter','latex'); grid on;
tit = title('2. Demanda absorbida vs $\beta$ para diferentes $\lambda$','interpreter','latex');
lg = legend('$\lambda = 5$','$\lambda = 5$ MIP','$\lambda = 6$','$\lambda = 6$ MIP','Location','best','interpreter','latex');


tit.FontSize = 9;
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');
lg.FontSize = 9;


subplot(223);
color = {'blue','red'};
for i = 1:length(lams)
    c = color(i);
    plot(betas,obj_val_relax(:,i),'-o','LineWidth',1.5,'Color',string(c));
    hold on;
    plot(betas,obj_val_MIP(:,i),'--o','LineWidth',1.5,'Color',string(c));
    hold on;
end

xl = xlabel('$\beta$','interpreter','latex'); yl = ylabel('valor objetivo','interpreter','latex'); grid on;
tit = title('3. Valor objetivo (MIP) vs $\beta$ para diferentes $\lambda$','interpreter','latex');
lg = legend('$\lambda = 5$','$\lambda = 5$ MIP','$\lambda = 6$','$\lambda = 6$ MIP','Location','best','interpreter','latex');

obj_gap = (obj_val_relax - obj_val_MIP)./obj_val_MIP;


tit.FontSize = 9;
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');
lg.FontSize = 9;


subplot(224);
color = {'blue','red'};
for i = 1:length(lams)
    c = color(i);
    semilogy(betas,time(:,i),'-o','LineWidth',1.5,'Color',string(c));
    hold on;
    semilogy(betas,time_MIP(:,i),'--o','LineWidth',1.5,'Color',string(c));
    hold on;
end

xl = xlabel('$\beta$','interpreter','latex'); yl = ylabel('tiempo [s]','interpreter','latex'); grid on;
tit = title('4. Tiempo de computo [s] vs $\beta$ para diferentes $\lambda$', 'interpreter','latex');
lg = legend('$\lambda = 5$','$\lambda = 5$ MIP','$\lambda = 6$','$\lambda = 6$ MIP','Location','northeast','interpreter','latex');

tit.FontSize = 9;
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');
lg.FontSize = 9;

saveas(gcf, 'comparativa_MIP.png'); % Guardar la figura en formato PNG




%% Tabla comparativa algoritmos
 
budgets = zeros(length(betas),length(lams));
budgets_MIP = budgets;
att_dem = zeros(length(betas),length(lams));
att_dem_MIP = att_dem;
nlinks = zeros(length(betas),length(lams));
nlinks_MIP = nlinks;
obj_val_relax = zeros(length(betas),length(lams));
obj_val_MIP = obj_val_relax;
time = zeros(length(betas),length(lams));
time_MIP = time;
close all;
n = 9;
figure('Position', [100, 100, 675, 450]);
for i = 1:length(betas)
    beta = betas(i);
    for j = 1:length(lams)
        lam = lams(j);
        %filename = sprintf('./results/betas/sol_beta=%d_lam=%d.mat',beta,lam);
       filename = sprintf('../red_9_nodos/uthopy_results/sol_beta=%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIP_sparsityaprim_budget=beta%d_lam=%d.mat',beta,lam);
         %filename = sprintf('./results/betas/sol_log2_beta_sparsityprim=%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIPmod_budget=beta%d_lam=%d.mat',beta,lam);
%         load(filename);
%         disp(['beta = ',num2str(beta),' lam = ',num2str(lam)]);
%         disp(['objs: pax = ',num2str(pax_obj),', op = ',num2str(op_obj),' obj = ', num2str(obj_val)]);
%         budgets(i,j) = budget;
%         att_dem(i,j) = sum(sum(f.*demand));
%         eps = 1e-3;
%         nlinks(i,j) = sum(sum(a_prim > eps));
%         obj_val_relax(i,j) = obj_val;
%         time(i,j) = comp_time;

        %filename = sprintf('./results/betas/sol_beta_sparsityprim=%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_log_beta_sparsityprim=%d_lam=%d.mat',beta,lam);
 %       filename = sprintf('../red_9_nodos/uthopy_results/sol_MIP_beta=%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIPnoLuis_sparsityaprim_budget=beta%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIPmod2_budget=beta%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIP_budget=beta%d_lam=%d.mat',beta,lam);
        %filename = sprintf('./results/betas/sol_MIP2iter_budget=beta%d_lam=%d.mat',beta,lam);
        load(filename);
        budgets_MIP(i,j) = budget;
        att_dem_MIP(i,j) = sum(sum(f.*demand));
        %nlinks_MIP(i,j) = sum(sum(a_bin));
        eps = 1e-3;
        nlinks_MIP(i,j) = sum(sum(a_prim > eps));
        obj_val_MIP(i,j) = obj_val;
        time_MIP(i,j) = comp_time;

    end
end




subplot(221);
color = {'blue','red'};
for i = 1:length(lams)
    c = color(i);
    plot(betas,nlinks(:,i),'-o','LineWidth',1.5,'Color',string(c));
    hold on;
    plot(betas,nlinks_MIP(:,i),'--o','LineWidth',1.5,'Color',string(c));
    hold on;
end

xl = xlabel('$\beta$','interpreter','latex'); yl = ylabel('arcos construidos','interpreter','latex'); grid on;
tit = title('1. Arcos construidos vs $\beta$ para diferentes $\lambda$','interpreter','latex');
lg = legend('$\lambda = 5$','$\lambda = 5$ MIP','$\lambda = 6$','$\lambda = 6$ MIP','Location','best','interpreter','latex');


tit.FontSize = 9;
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');
lg.FontSize = 9;

subplot(222);
color = {'blue','red'};
for i = 1:length(lams)
    c = color(i);
    plot(betas,100.*att_dem(:,i)./total_demand,'-o','LineWidth',1.5,'Color',string(c));
    hold on;
    plot(betas,100.*att_dem_MIP(:,i)./total_demand,'--o','LineWidth',1.5,'Color',string(c));
    hold on;
end

xl = xlabel('$\beta$','interpreter','latex'); yl = ylabel('demanda absorbida [\%]','interpreter','latex'); grid on;
tit = title('2. Demanda absorbida vs $\beta$ para diferentes $\lambda$','interpreter','latex');
lg = legend('$\lambda = 5$','$\lambda = 5$ MIP','$\lambda = 6$','$\lambda = 6$ MIP','Location','best','interpreter','latex');


tit.FontSize = 9;
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');
lg.FontSize = 9;


subplot(223);
color = {'blue','red'};
for i = 1:length(lams)
    c = color(i);
    plot(betas,obj_val_relax(:,i),'-o','LineWidth',1.5,'Color',string(c));
    hold on;
    plot(betas,obj_val_MIP(:,i),'--o','LineWidth',1.5,'Color',string(c));
    hold on;
end

xl = xlabel('$\beta$','interpreter','latex'); yl = ylabel('valor objetivo','interpreter','latex'); grid on;
tit = title('3. Valor objetivo (MIP) vs $\beta$ para diferentes $\lambda$','interpreter','latex');
lg = legend('$\lambda = 5$','$\lambda = 5$ MIP','$\lambda = 6$','$\lambda = 6$ MIP','Location','best','interpreter','latex');

obj_gap = (obj_val_relax - obj_val_MIP)./obj_val_MIP;


tit.FontSize = 9;
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');
lg.FontSize = 9;


subplot(224);
color = {'blue','red'};
for i = 1:length(lams)
    c = color(i);
    semilogy(betas,time(:,i),'-o','LineWidth',1.5,'Color',string(c));
    hold on;
    semilogy(betas,time_MIP(:,i),'--o','LineWidth',1.5,'Color',string(c));
    hold on;
end

xl = xlabel('$\beta$','interpreter','latex'); yl = ylabel('tiempo [s]','interpreter','latex'); grid on;
tit = title('4. Tiempo de computo [s] vs $\beta$ para diferentes $\lambda$', 'interpreter','latex');
lg = legend('$\lambda = 5$','$\lambda = 5$ MIP','$\lambda = 6$','$\lambda = 6$ MIP','Location','northeast','interpreter','latex');

tit.FontSize = 9;
set(gca, 'FontSize', 9);
set(gca, 'TickLabelInterpreter', 'latex');
lg.FontSize = 9;

%% Funciones

function [obj_val,pax_obj,op_obj] = get_obj_val(alfa, op_link_cost,...
    congestion_coef_links, ...
    congestion_coef_stations,travel_time,prices,alt_time,alt_price,a_prim,delta_a, ...
    s_prim,delta_s,fij,f,fext,demand,dm_pax,dm_op)
    n = 9;
    pax_obj = 0;
    op_obj = 0;
    eps = 1e-3;
    op_obj = op_obj + 1e-6*(sum(sum(op_link_cost.*a_prim))); %operational costs
    for i=1:n
        if s_prim(i) > eps
            pax_obj = pax_obj + 1e-6*inv_pos(congestion_coef_stations(i)*delta_s(i) + eps);
        end

        for j=1:n
            if a_prim(i,j) > eps
                pax_obj = pax_obj + 1e-6*inv_pos(congestion_coef_links(i,j)*delta_a(i,j) + eps);
            end
        end
    end
    for o=1:n
        for d=1:n
            for i=1:n
                for j=1:n
                    pax_obj = pax_obj + 1e-6*(demand(o,d).*(travel_time(i,j)+prices(i,j)).*fij(i,j,o,d));
                end
            end
        end
    end
    pax_obj = pax_obj + 1e-6*(sum(sum(demand.*(alt_time+alt_price).*fext)));
    pax_obj = pax_obj + 1e-6*(sum(sum(demand.*(-entr(f) - f))));
    pax_obj = pax_obj + 1e-6*(sum(sum(demand.*(-entr(fext) - fext))));
    obj_val = pax_obj*alfa/dm_pax + op_obj*(1-alfa)/dm_op;
end




function [n,link_cost,station_cost,link_capacity_slope,...
    station_capacity_slope,demand,prices,...
    load_factor,op_link_cost,congestion_coef_stations,...
    congestion_coef_links,travel_time,alt_time,alt_price,a_nom,tau,eta,...
    a_max,candidates] = parameters_9node_network()

    n = 9;
    
    %candidates to construct a link for each neighbor
    candidates = {[2,3,9],[1,3,4],[9,1,2,4,5],[2,3,5,6,8],[3,4,6,7],[4,5,7,8],[5,6],[4,6],[1,3]};
    
    %cost of using the alternative network for each o-d pair
    alt_cost = [0,1.6,0.8,2,1.6,2.5,3,2.5,0.8; 
                2,0,0.9,1.2,1.5,2.5,2.7,2.4,1.8; 
                1.5,1.4,0,1.3,0.9,2,1.6,2.3,0.9; 
                1.9,2,1.9,0,1.8,2,1.9,1.2,2; 
                3,1.5,2,2,0,1.5,1.1,1.8,1.7; 
                2.1,2.7,2.2,1,1.5,0,0.9,0.9,2.9; 
                2.8,2.3,1.5,1.8,0.9,0.8,0,1.3,2.1; 
                2.8,2.2,2,1.1,1.5,0.8,1.9,0,0.3; 
                1,1.5,1.1,2.7,1.9,1.8,2.4,3,0];
    
    %fixed cost for constructing links
    link_cost = (1e6/(25*365.25)).*[0,1.7,2.7,0,0,0,0,0,2.9; 
                 1.7,0,2.1,3,0,0,0,0,0; 
                 2.7,2.1,0,2.6,1.7,0,0,0,2.5; 
                 0,3,2.6,0,2.8,2.4,0,3.2,0; 
                 0,0,1.7,2.8,0,1.9,3,0,0; 
                 0,0,0,2.4,1.9,0,2.7,2.8,0; 
                 0,0,0,0,3,2.7,0,0,0; 
                 0,0,0,3.2,0,2.8,0,0,0; 
                 2.9,0,2.5,0,0,0,0,0,0];
    link_cost (link_cost ==0) = 1e4;
    
    %fixed cost for constructing stations
    station_cost = (1e6/(25*365.25)).*[2, 3, 2.2, 3, 2.5, 1.3, 2.8, 2.2, 3.1];
    
    link_capacity_slope = 0.04.*link_cost; 
    station_capacity_slope = 0.04.*station_cost;
    
    %demand between od pairs
    demand = 1e3.*[0,9,26,19,13,12,13,8,11;
              11,0,14,26,7,18,3,6,12;
              30,19,0,30,24,8,15,12,5;
              21,9,11,0,22,16,25,21,23;
              14,14,8,9,0,20,16,22,21;
              26,1,22,24,13,0,16,14,12;
              8,6,9,23,6,13,0,11,11;
              9,2,14,20,18,16,11,0,4;
              8,7,11,22,27,17,8,12,0];
    
    distance = 10000 * ones(n, n); % Distances between arcs
    
    for i = 1:n
        distance(i, i) = 0;
    end
    
    distance(1, 2) = 0.75;
    distance(1, 3) = 0.7;
    distance(1, 9) = 0.9;
    
    distance(2, 3) = 0.6;
    distance(2, 4) = 1.1;
    
    distance(3, 4) = 1.1;
    distance(3, 5) = 0.5;
    distance(3, 9) = 0.7;
    
    distance(4,5) = 0.8;
    distance(4,6) = 0.7;
    distance(4,8) = 0.8;
    
    distance(5,6) = 0.5;
    distance(5,7) = 0.7;
    
    distance(6,7) = 0.5;
    distance(6,8) = 0.4;
    
    for i = 1:n
        for j = i+1:n
            distance(j, i) = distance(i, j); % Distances are symmetric
        end
    end
    
    %Load factor on stations
    load_factor = 0.25 .* ones(1, n);
    
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
    eta = 0.25;
    a_max = 1e9;
    eps = 1e-3;
end
