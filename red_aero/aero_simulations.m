close all;
clear all;
clc;
[n,link_cost,station_cost,...
station_capacity_slope,demand,prices,...
op_link_cost,congestion_coef_airline, congestion_coef_airport,...
travel_time,alt_time,alt_price,a_nom,tau,sigma,...
a_max,trans_time,s_max,station_hub_cost,distance] = parameters_aero_network();




%% Resolvemos los problemas por separado para calcular el punto de utopía
%Problema del operador:
%op_obj = operation_costs
%obj_val = 0
%Problema de los pasajeros:
%pax_obj = (distances + prices) + entropies + congestion
%obj_val. Se calcula para un presupuesto dado. En este caso daremos un
%valor a beta, calculamos el presupuest obtenido y lo mantenemos todo el
%tiempo.

betas = [8,10,15,20];
lams = [0.4,1,3,5,7,10,15];

betas = [0,0.5,1,2,3];
lams = [1,1.5,2,3];
betas = [7,10,12];


betas = [5,7,10];
betas = 7.1:0.1:7.8; %optimistic, same bud
betas = 6.2:0.1:6.9;
% lams = [3,5,7];
% lams = [10];
% lams = [12];
% lams = [15];
% lams = [20];
% lams = [17];
% lams = [16];
lams = [5,10,20];
% lams = [20];
% betas = 4:0.2:5.4;
% betas = 3.2:0.1:3.9; %same bud, lam 10
% betas = [3.2:0.2:4.6]; %same bud, lam 5
% betas = 4.7:0.2:6.3; %same bud, lam 20

%betas a simular despues con alfa = 1: 0.5 hasta 2, de 0.1 en 0.1), subir
%threshold para matar arcos hasta 1 o 2 (preguntar), los mas bajos salen 16, pero en la iter 9 había 220 mayores que 1, y 376 al quitar coste fijo con 0.1 de threshold.
budgets = zeros(length(betas),1);
cvx_solver_settings -clearall
cvx_solver mosek
cvx_precision high
cvx_save_prefs 
num_workers = 8;
% Inicia el parpool para parfor
parpool('local', num_workers); % num_workers es el número de trabajadores a utilizar
parfor bb=1:length(betas) 
    for ll=1:length(lams)
        eps = 1e-3;
        alfa = 0.5;
        lam = lams(ll);
        beta = betas(bb);
        dm_pax = 1; %calcular con análisis de utopía
        dm_op = 1; %calcular con análisis de utopía
        [s,s_h,s_prim,s_h_prim,delta_s,delta_s_prim,delta_s_h,a,f,fext,fij,comp_time,budget,obj_val,...
        pax_obj,op_obj] = compute_sim(lam,beta,alfa,dm_pax,dm_op);
        disp(['nlinks =',num2str(sum(sum(a > 1)))]);

        
       % disp(['budget = ',num2str(budget)]);
     %   budgets(bb,ll) = budget;
        
        disp(['obj_val = ',num2str(obj_val),', pax_obj = ',num2str(pax_obj), ...
            ', op_obj = ',num2str(op_obj)]);

    end
end
% Cierra el parpool
delete(gcp);
%%
close all;
clear colores;
cvx_begin
cvx_end
betas = [10,15,20];
lams = [10,15,18];
%para limitar amax beta 10 lam 20, beta 15 lam 10
betas = [10,15];
lams = [1,3,5];
lams = [0.4,0.42,0.5];
lams = [0.4];

betas = [10,15,20];
betas = [8,10,20];
lams = [0.4,1,3];

betas = [8];
lams = [1];

betas = [8,10,15,20];
lams = [0.4,1,3,5,7,10,15];
lams = [10];
betas = [10];

betas = [0,0.5,1,2,3,5,7,10,12];
betas = [1,3,5];
betas = [7,10,12];
betas  = [1,3,5,7,10,12];
betas = [1,7,10];
lams = [1,1.5,2,3];
betas = [7];
lams = [2];
%betas = [5];
%lams = [3];


% con logit 0.5
 % betas = [10,15,20];
 % lams = [7,10,15];
 betas = [10];
 lams = [7];


 %con logit 0.2
% betas = [5,7,10];
% lams = [3,5,7];
betas = [7];
lams = [5];
% 
% 
% %con logit 0.3
betas = [5,7,10];
betas = [7];
betas = [5];
lams = [3,5,7,10,20];
lams = [5,10,15,16,17];
lams = [5,10,20];
lams = [5,10,20];
betas = [4:0.1:4.7,5];
betas = [5,7,10];
betas = [5,7,10];

betas = [7];
lams = [10];


%slim, lam 5. beta = 3.6
%slim, lam 10. beta =3.6
%sli, lam 20. beta = 5.9
%betas = 3.2:0.1:3.9; %same bud, lam 10
% betas = 7.1:0.1:7.8; %optimistic, same bud
% betas = 6.2:0.1:6.9; %pesimistic, same bud

%betas = 7.1:0.1:
%  lams= [5,10,20];
%  lams = [10]; betas = [7];
  % lams = [20];
 %  betas = 4.7:0.2:6.3;
  % betas = [3.2:0.2:4.6];
   % lams = [20];
   % betas = [3.2:0.2:4.6]; %same bud, lam 5
 %    betas = [5,5.8,5.9,6,6.1,6.2,6.3,6.4];
 % betas = 4:0.2:5.4;
 % betas = 3.2:0.1:3.9;
% betas = 5;
clc;
alfa = 0.5;
dm_pax = 1;
dm_op = 1;
coef_logit = 0.3;
[n,link_cost,station_cost,...
    station_capacity_slope,demand,prices,...
    op_link_cost,congestion_coef_airline, congestion_coef_airport,...
    travel_time,alt_time,alt_price,a_nom,tau,sigma,...
    a_max,trans_time,s_max,station_hub_cost] = parameters_aero_network();
CAB_data = readtable('./CAB_data.xlsx');
coor_x = table2array(CAB_data(1:25,1));
coor_y = table2array(CAB_data(1:25,2));
counter = 1;
for bb=1:length(betas)
    for ll=1:length(lams)
        
        beta_or = betas(bb);
       %  beta_or = 3.6;
         lam = lams(ll);
        % 
        % if lam == 20
        %     beta_or = 5.9;
        % end

        filename = sprintf('./aero_results/prelim_sol_beta=%d_lam=%d.mat',beta_or,lam);
        filename = sprintf('./aero_results/prelim_symmetric_sol_beta=%d_lam=%d.mat',beta_or,lam);
       filename = sprintf('./aero_results/prelim_symmetric_logcoef03_sol_beta=%d_lam=%d.mat',beta_or,lam);
      % filename = sprintf('./aero_results/prelim_symmetric_logcoef03_slim_sol_beta=%d_lam=%d.mat',beta_or,lam);
      % filename = sprintf('./aero_results/prelim_symmetric_logcoef03_slim_samebud_sol_beta=%d_lam=%d.mat',beta_or,lam);
      %  filename = sprintf('./aero_results/prelim_symmetric_logcoef03_alim_sol_beta=%d_lam=%d.mat',beta_or,lam);
      %  filename = sprintf('./aero_results/prelim_symmetric_logcoef03_cprices_sol_beta=%d_lam=%d.mat',beta_or,lam);
     %   filename = sprintf('./aero_results/prelim_symmetric_logcoef03_pesimistic_sol_beta=%d_lam=%d.mat',beta_or,lam);
    %  filename = sprintf('./aero_results/prelim_symmetric_logcoef03_optimistic_sol_beta=%d_lam=%d.mat',beta_or,lam);
     %   filename = sprintf('./aero_results/prelim_symmetric_slim_sol_beta=%d_lam=%d.mat',beta_or,lam);
     %   filename = sprintf('./aero_results/prelim_symmetric_alim_sol_beta=%d_lam=%d.mat',beta_or,lam);
       % filename = sprintf('./aero_results/prelim_alim_sol_beta=%d_lam=%d.mat',beta_or,lam);
      %  filename = sprintf('./aero_results/prelim_optimistic_sol_beta=%d_lam=%d.mat',beta_or,lam);
     %   filename = sprintf('./aero_results/prelim_pesimistic_sol_beta=%d_lam=%d.mat',beta_or,lam); 
       % filename = sprintf('./aero_results/prelim_pricing_sol_beta=%d_lam=%d.mat',beta_or,lam);
      %  filename = sprintf('./aero_results/prelim_slim_sol_beta=%d_lam=%d.mat',beta_or,lam);
        load(filename);
       % filename = sprintf('./aero_results/prelim_pesimistic_sol_beta=%d_lam=%d.mat',beta_or,lam);
      %  filename = sprintf('./aero_results/prelim_02dem_sol_beta=%d_lam=%d.mat',beta_or,lam);
       % filename = sprintf('./aero_results/prelim_0dem_sol_beta=%d_lam=%d.mat',beta_or,lam);
       % filename = sprintf('./aero_results/prelim_slim_sol_beta=%d_lam=%d.mat',beta_or,lam);
        [obj_val,pax_obj,op_obj] = get_obj_val(alfa,...
    a, s_h_prim,s_prim,delta_s,delta_s_h,delta_s_prim,fij,f,fext,dm_pax,dm_op,coef_logit);
        disp(['beta = ',num2str(beta_or), ', lam = ',num2str(lam)]);

     
        budget = get_budget(s_prim,s_h_prim,a,n,...
    station_cost,station_hub_cost,station_capacity_slope,link_cost,lam);
        disp(['budget = ',num2str(budget)]);
        att_dem = sum(sum(demand.*f))/sum(sum(demand));
        disp(['att dem = ',num2str(att_dem)]);
        disp(['nhubs = ',num2str(sum(s_h > 0))]);
        disp(['nairports = ',num2str(sum(s + s_h > 0))]);
         disp(['nlinks =',num2str(sum(sum(a > 0.01)))]);
        

        ingresos = 0;
        rask = zeros(n);
        rask_t = 0;
        rev = 0;
        ask = 0;
        cask_num = 0;
        cask_den = 0;
        distance = (travel_time-0.6).*450;
        for o=1:n
            for d=1:n
                ingresos = ingresos + f(o,d)*prices(o,d)*demand(o,d);
                if f(o,d) > 0.001
                    rask(o,d) = prices(o,d)*demand(o,d)*f(o,d)./(demand(o,d)*sum(sum(squeeze(fij(:,:,o,d)).*distance)));
                    rev = rev + prices(o,d)*demand(o,d)*f(o,d);
                    ask = ask  + demand(o,d)*sum(sum(squeeze(fij(:,:,o,d)).*distance));
                    
                end
            end
        end
        for i=1:n
            for j=1:n
                cask_num = cask_num + op_link_cost(i,j)*a(i,j);
                cask_den = cask_den + a_nom*a(i,j)*distance(i,j);
            end
        end
        gastos = sum(sum(op_link_cost.*a));
        beneficios = ingresos - gastos;
        disp(['beneficios= ',num2str(beneficios)]);
        disp(['RASK = ',num2str(mean(mean(rev/ask)))]);
        disp(['CASK = ',num2str(mean(mean(cask_num/cask_den)))]);
        disp(['tiempo = ',num2str(comp_time)]);
        

      %  figure;
        subplot(length(betas),length(lams),counter);
        counter = counter + 1;
        a(a < 0.01) = 0;
        g = graph(a);
        colormap parula
        colors = colormap;
     %   numColors = size(colors,1);
        % scaledValues = round(1+(numColors-1)*(g.Edges.Weight-min(g.Edges.Weight))/(max(g.Edges.Weight)-min(g.Edges.Weight)));
        % ee = 0;
        
        [sorted_edges,sorted_edges_pos] = sort(g.Edges.Weight,'ascend');
        scaled_pos = round(sorted_edges_pos.*(length(colors)/length(sorted_edges_pos)));
        % scaled_pos_q = scaled_pos;
        %  % scaled_pos_q(scaled_pos > quantile(scaled_pos,0.75)) = round(max(scaled_pos));
        %  % scaled_pos_q(scaled_pos < quantile(scaled_pos,0.75)) = round(quantile(scaled_pos,0.66));
        %  % scaled_pos_q(scaled_pos < quantile(scaled_pos,0.5)) = round(quantile(scaled_pos,0.33));
        %  % scaled_pos_q(scaled_pos < quantile(scaled_pos,0.25)) = round(quantile(scaled_pos,0.01));

        
        
   %     link_pos = 


        colores_edg = colors( round(1 + g.Edges.Weight .* length(colors)/max(g.Edges.Weight+1)) ,:);
        % for ei=1:n
        %     if sum(sum(a(ei,:) > 0.1)) > 1
        %         ee = ee + 1;
        %         colores_edg(ee,:) = colors(scaledValues(ee),:);
        %        % highlight(h,g.Edges.EndNodes(ee,:),'EdgeColor',colors(scaledValues(ee),:)); 
        %     end
        % end


        colores = zeros(n,3);
        for i=1:n
            if s_h(i) > 0.1
                colores(i,:) = [0.6350 0.0780 0.1840];
            else
                colores(i,:) = [0 0 0];%[0 0.4470 0.7410];
            end
        end
        h = plot(g,'XData',coor_x-mean(coor_x),'YData',coor_y-mean(coor_y), ...
            'MarkerSize',0.7*(s_h+s).^0.5 +1e-2,'LineWidth', ...
            0.1.*g.Edges.Weight,'NodeColor',colores,'EdgeColor',colores_edg,'EdgeAlpha',0.7,'NodeFontSize',8);
        xticks([]); yticks([]); 
       % colormap turbo;
        caxis([min(g.Edges.Weight),max(g.Edges.Weight)]);
        colorbar




        hold on
        I = imread('./us_map.jpg'); 
        h = image(0.93*xlim,-ylim,I); 
        uistack(h,'bottom');
        tit = ['\lambda = ',num2str(lam),', \beta = ',num2str(beta_or),', ',num2str(sum(s_h > 0)),' hubs',...
            ', demanda atendida = ',num2str(att_dem*100,2),'%',' , budg = ',num2str(budget,2)];

        tit = ['\lambda = ',num2str(lam),', \beta = ',num2str(beta_or),' , ', num2str(length(g.Edges.Weight)*2),' enlaces',...
            ', demanda atendida = ',num2str(att_dem*100,2),'%'];
        title(tit,'FontSize',12);

        pax = zeros(n);
        for i=1:n
            cons_1_gap(i) = sum(a(:,i)) - s(i) - s_h(i);
            for j=1:n
                cons_2_gap(i,j) = a(i,j) - a(j,i);
                pax = squeeze(sum(sum(squeeze(permute(fij(i,j,:,:),[3 4 1 2]).*demand))));
                cons_3_gap(i,j) = pax/(a_nom*tau) - a(i,j);
                
            end
            cons_4_gap(i) =  max(sum(sum(squeeze(permute(sum(fij(i,:,:,:),2),[3,4,1,2])).*demand)),sum(sum(squeeze(permute(sum(fij(:,i,:,:),1),[3,4,1,2])).*demand)))/(a_nom*tau) - s_prim(i) - s_h_prim(i);
        end
      %  disp(min(min(cons_4_gap)));
       % disp(min(cons_3_gap));

    end
end
set(gcf, 'Position', [100, 100, 935*2, 587*2.2]);
set(gcf, 'Position', [100, 100, 935*2, 587]);
figurename = sprintf('./figures/topologia_slim_samebud.png');
%saveas(gcf, figurename);

%% Calcular porcentaje de uso
pax = zeros(n);
for i=1:n
    cons_1_gap(i) = sum(a(:,i)) - s(i) - s_h(i);
    for j=1:n
        cons_2_gap(i,j) = a(i,j) - a(j,i);
        pax = squeeze(sum(sum(squeeze(permute(fij(i,j,:,:),[3 4 1 2]).*demand))));
        cons_3_gap(i,j) = pax/(a_nom*tau) - a(i,j);
        
    end
    cons_4_gap(i) =  max(sum(sum(squeeze(permute(sum(fij(i,:,:,:),2),[3,4,1,2])).*demand)),sum(sum(squeeze(permute(sum(fij(:,i,:,:),1),[3,4,1,2])).*demand)))/(a_nom*tau) - s(i) - s_h(i);
end
   %%

for i=1:n
    for j=1:n

        pax(i,j) = squeeze(sum(sum(squeeze(permute(fij(i,j,:,:),[3 4 1 2]).*demand))));
       % pax_sal(i) = sum(sum(squeeze(permute(sum(fij(i,:,:,:),2),[3,4,1,2])).*demand));
       % pax_ent(i) = sum(sum(squeeze(permute(sum(fij(:,i,:,:),1),[3,4,1,2])).*demand));
       % arcos_uso(i) = sum(a(:,i));
       % pax_ent(i) = pax_ent(i) + sum(fij(:,i,o,d),1)*demand(o,d);
    end
end
%% Calcular demanda atendida en vuelo directo

for nodo=1:n
    total_nodo = sum(sum(squeeze(permute(sum(fij(nodo,:,:,:),2),[3,4,1,2])).*demand)) + sum(sum(squeeze(permute(sum(fij(:,nodo,:,:),1),[3,4,1,2])).*demand));
    propio_nodo = sum(sum(squeeze(permute(sum(fij(nodo,:,nodo,:),2),[3,4,1,2])).*demand(nodo,:))) + sum(sum(squeeze(permute(sum(fij(:,nodo,:,nodo),1),[3,4,1,2])).*demand(:,nodo)));
    prop_directo(nodo) = propio_nodo/total_nodo;
end
figure(1);
stem(100.*(1-prop_directo),'LineWidth',1.5);
ax = gca;
names = {'Atlanta','Baltimore','Boston','Chicago','Cincinatti',...
    'Cleveland','Dallas','Denver','Detroit','Houston','Kansas','Los Angeles',...
    'Memphis','Miami','Minneapolis','Nueva Orleans','Nueva York',...
    'Filadelfia','Phoenix','Pittsburgh','St. Luis','San Francisco',...
    'Seattle','Tampa','Washington DC'};
xticks(1:25)
xticklabels(names);
yticks(0:5:20);
ylim([0 20]);
xlim([0.75 25.25]);
title('Vuelos en escala por ciudad','FontSize',20);
yticklabels({'0 %','5 %','10 %','15 %','20 %'});
ylabel('[%] de vuelos totales');
%yl = ylabel('% de vuelos de conexión')
 ax.XTickLabelRotation = 60;
 ax.FontSize = 12;

%%
for nodo=1:25
    total_nodo =  sum(sum(squeeze(permute(sum(fij(:,nodo,:,:),1),[3,4,1,2])).*demand));
    nohubs_nodo =  sum(sum(squeeze(permute(sum(fij(:,nodo,[1:3,5:11,13:16,18:21,23:25],:),1),[3,4,1,2])).*demand([1:3,5:11,13:16,18:21,23:25],:)));
    prop_nohubs(nodo) = nohubs_nodo/total_nodo;
end
close all;
stem(prop_nohubs)

%% Calcular beneficios
ingresos = 0;
rask = zeros(n);
rask_t = 0;
rev = 0;
ask = 0;
cask_num = 0;
cask_den = 0;
distance = (travel_time-0.6).*450;
for o=1:n
    for d=1:n
        ingresos = ingresos + f(o,d)*prices(o,d)*demand(o,d);
        if f(o,d) > 0.001
            rask(o,d) = prices(o,d)*demand(o,d)*f(o,d)./(demand(o,d)*sum(sum(squeeze(fij(:,:,o,d)).*distance)));
            rev = rev + prices(o,d)*demand(o,d)*f(o,d);
            ask = ask  + demand(o,d)*sum(sum(squeeze(fij(:,:,o,d)).*distance));
            
        end
    end
end
for i=1:n
    for j=1:n
        cask_num = cask_num + op_link_cost(i,j)*a(i,j);
        cask_den = cask_den + a_nom*a(i,j)*distance(i,j);
    end
end
gastos = sum(sum(op_link_cost.*a));
beneficios = ingresos - gastos;
rev/ask
cask_num/cask_den



%% Functions
function budget = get_budget(s_prim,s_h_prim,a,n,...
    station_cost,station_hub_cost,station_capacity_slope,link_cost,lam)
    budget = 0;
    sigma = 0.18;
    budget = budget + sum(station_capacity_slope'.*(s_prim + s_h_prim));
    for i=1:n
        if (s_prim(i) > 0.1*sigma) | (s_h_prim(i) > 0.1*sigma)
            budget = budget + station_cost(i);
        end
        if s_h_prim(i) > 0.1
            budget = budget + lam*station_hub_cost(i);
        end
        for j=1:n
            if a(i,j) > 0.1
                budget = budget + link_cost(i,j);
            end
        end
    end
end
function [obj_val,pax_obj,op_obj] = get_obj_val(alfa,...
    a, s_h_prim,s_prim,delta_s,delta_s_h,delta_s_prim,fij,f,fext,dm_pax,dm_op,coef_logit)
    n = 25;
    [n,link_cost,station_cost,...
    station_capacity_slope,demand,prices,...
    op_link_cost,congestion_coef_airline, congestion_coef_airport,...
    travel_time,alt_time,alt_price,a_nom,tau,sigma,...
    a_max,trans_time,s_max,station_hub_cost] = parameters_aero_network();          
    eps = 1e-3;

    op_obj = 0;
    pax_obj = 0;
    op_obj = op_obj + 1e-6*(sum(sum(op_link_cost.*a))); %operation costs
    
    f = max(f,0);
    fext = max(fext,0);

    for i=1:n
        if s_prim(i) > 0
            pax_obj = pax_obj + 1e-6*(sum(inv_pos(congestion_coef_airline.*delta_s(i) + eps))); %congestion costs
        end
        if s_h_prim(i) > 0
            pax_obj = pax_obj + 1e-6*(sum(inv_pos(congestion_coef_airline.*delta_s_h(i) + eps)));
        end
        if (s_prim(i) > 0) | (s_h_prim(i) > 0)
            pax_obj = pax_obj + 1e-6*(sum(inv_pos(congestion_coef_airport.*delta_s_prim(i) + eps)));
        end
    end
    for o=1:n
        for d=1:n
            pax_obj = pax_obj + 1e-6*(demand(o,d).*prices(o,d).*sum(sum((travel_time+trans_time(:,:,o,d)).*fij(:,:,o,d).*coef_logit))); 
        end
    end
    pax_obj = pax_obj + 1e-6*(sum(sum(demand.*prices.*(alt_time+0.01.*alt_price).*fext.*coef_logit)));
    pax_obj = pax_obj + 1e-6*(sum(sum(demand.*prices.*(-entr(f) - f)))) + 1e-6*(sum(sum(demand.*prices.*0.01.*prices.*f.*coef_logit)));
    pax_obj = pax_obj + 1e-6*(sum(sum(demand.*prices.*(-5.*entr(0.2.*fext) - fext))));
    
    obj_val = (alfa/(dm_pax))*pax_obj + ((1-alfa)/(dm_op))*op_obj;
end

function [s,s_h,s_prim,s_h_prim,delta_s,delta_s_prim,delta_s_h,a,f,fext,fij,comp_time,budget,obj_val,...
    pax_obj,op_obj] = compute_sim(lam,beta_or,alfa,dm_pax,dm_op)

    
    [n,link_cost,station_cost,...
    station_capacity_slope,demand,prices,...
    op_link_cost,congestion_coef_airline, congestion_coef_airport,...
    travel_time,alt_time,alt_price,a_nom,tau,sigma,...
    a_max,trans_time,s_max,station_hub_cost] = parameters_aero_network();
    niters = 10;           
    eps = 1e-3;

    %  beta_oo = 7;
    %  filename = sprintf('./aero_results/prelim_symmetric_logcoef03_sol_beta=%d_lam=%d.mat',beta_oo,lam);
    %   load(filename);
    % s_max = 1e5.*ones(n,1);
    % [val arg] = sort(s_h_prim);
    %  for i=0:2
    %     s_max(arg(n-i)) = s_h_prim(arg(n-i))*0.6;
    %  end

    % filename = sprintf('./aero_results/prelim_lastiter_base_sol_beta=%d_lam=%d.mat',beta_or,lam);
    % load(filename);
    %a_max = sum(sum(a)).*0.8;

    a_prev = 1e4.*ones(n);
    s_prev= 1e4.*ones(n,1);
    s_h_prev = s_prev;
    disp(['beta = ',num2str(beta_or),', lam = ',num2str(lam)]);
    tic;
    for iter=1:niters
        cvx_begin quiet
            variable s(n)
            variable s_h(n)
            variable s_h_prim(n)
            variable s_prim(n)
            variable delta_s(n)
            variable delta_s_h(n)
            variable delta_s_prim(n)
            variable a(n,n)
            variable f(n,n)
            variable fext(n,n)
            variable fij(n,n,n,n)
            coef_logit = 0.3;
            op_obj = 0;
            pax_obj = 0;
            bud_obj = 0;
            bud_obj = bud_obj + 1e-2*sum(station_capacity_slope'.*s_prim);
            bud_obj = bud_obj + 1e-2*sum(station_capacity_slope'.*s_h_prim);
            op_obj = op_obj + 1e-2*(sum(sum(op_link_cost.*a))); %operation costs
            if iter < niters
                pax_obj = pax_obj + 1e-2*(sum(inv_pos(congestion_coef_airline.*delta_s + eps))); %congestion costs
                pax_obj = pax_obj + 1e-2*(sum(inv_pos(congestion_coef_airport.*delta_s_prim + eps)));
                pax_obj = pax_obj + 1e-2*(sum(inv_pos(congestion_coef_airline.*delta_s_h + eps)));
                bud_obj = bud_obj + 1e-2*sum(sum((link_cost.*a.*(1./(a_prev+eps))))) + 1e-2*sum((station_cost'.*(s_prim+s_h_prim).*(1./(s_prev+s_h_prev+eps)))); %fixed construction costs
                bud_obj = bud_obj + 1e-2*sum(((lam*station_hub_cost'-station_cost').*s_h_prim.*(1./(s_h_prev+eps)))); %fixed sparsity costs
            end
    
            for o=1:n
                for d=1:n
                    pax_obj = pax_obj + 1e-2*(demand(o,d).*prices(o,d).*sum(sum((travel_time+trans_time(:,:,o,d)).*fij(:,:,o,d).*coef_logit))); 
                end
            end
            pax_obj = pax_obj + 1e-2*(sum(sum(demand.*prices.*(alt_time+0.01.*alt_price).*fext.*coef_logit)));
            pax_obj = pax_obj + 1e-2*(sum(sum(demand.*prices.*(-entr(f) - f)))) + 1e-2*(sum(sum(demand.*prices.*0.01.*prices.*f.*coef_logit)));
            pax_obj = pax_obj + 1e-2*(sum(sum(demand.*prices.*(-5.*entr(0.2.*fext) - fext))));
    
    
            if iter == niters
                for i=1:n
                    if s_prev(i) >= 0.1
                        pax_obj = pax_obj + 1e-2*(sum(inv_pos(congestion_coef_airline.*delta_s(i) + eps))); %congestion costs
                    end
                    if s_h_prev(i) >= 0.1
                        pax_obj = pax_obj + 1e-2*(sum(inv_pos(congestion_coef_airline.*delta_s_h(i) + eps))); %congestion costs
                    end
                    if (s_prev(i) >= 0.1) | (s_h_prev(i) >= 0.1)
                        pax_obj = pax_obj + 1e-2*(sum(inv_pos(congestion_coef_airport.*delta_s_prim(i) + eps)));
                    end
                 end
            end
            obj = beta_or*bud_obj + (alfa/(dm_pax))*pax_obj + ((1-alfa)/(dm_op))*op_obj;
            minimize obj
            % constraints
            s >= 0;
            s_prim >= 0;
            delta_s >= 0;
            s_h >= 0;
            s_h_prim >= 0;
            delta_s_h >= 0;
            a >= 0;
            f >= 0;
            f <= 1;
            fij >= 0;
            fij <= 1;
            fext >= 0;
            fext <= 1;
            s_prim == s + delta_s;
            s_h_prim == s_h + delta_s_h;
            s_max == s_prim + delta_s_prim + s_h_prim;

            for i=1:n
                sum(a(:,i)) <= s(i) + s_h(i);
                for j=1:n
                    a(i,j) == a(j,i);
                end
            end
    
            for i=1:n
                for j=1:n
                    squeeze(sum(sum(squeeze(permute(fij(i,j,:,:),[3 4 1 2]).*demand)))) <= a(i,j).*a_nom.*tau; %multiplicar por demanda
                   % sum(sum(squeeze(permute(fij(i,j,i,:),[3 4 1 2])).*demand(i,:))) <= a(i,j).*a_nom.*tau;
                end
            end

            for i=1:n
                 sigma*sum(sum(squeeze(permute(sum(fij(i,:,[1:(i-1),(i+1):n],:),2),[3,4,1,2])).*demand([1:(i-1),(i+1):n],:)))/a_nom <= s_h(i); %salidas hub
                 sigma*sum(sum(squeeze(permute(sum(fij(i,:,:,:),2),[3,4,1,2])).*demand))/a_nom <= s_h(i) + s(i); %salidas
            end

            for j=1:n
                sigma*sum(sum(squeeze(permute(sum(fij(:,j,:,[1:(j-1),(j+1):n]),1),[3,4,1,2])).*demand(:,[1:(j-1),(j+1):n])))/a_nom <= s_h(j); %entradas hub
                sigma*sum(sum(squeeze(permute(sum(fij(:,j,:,:),1),[3,4,1,2])).*demand))/a_nom <= s_h(j) + s(j); %entradas
            end

            
            for o=1:n
                for d=1:n
                    demand(o,d)*fij(o,d,o,d)*sigma/a_nom <= s_h(o) + s_h(d); %vuelos directos
                end
            end

            sum(sum(a)) <= a_max;
            for o=1:n
                for d=1:n
                    sum(fij(o,:,o,d)) == f(o,d);
                end
            end
            for i=1:n
                a(i,i) == 0;
            end
            
            for o=1:n
                squeeze(sum(fij(o,:,o,[1:(o-1),(o+1):n]),2)) - squeeze(sum(permute(fij(:,o,o,[1:(o-1),(o+1):n]),[2,1,3,4]),2)) == transpose(1 - fext(o,[1:(o-1),(o+1):n])); 
            end
            for d=1:n
                squeeze(sum(fij(d,:,[1:(d-1),(d+1):n],d),2)) - squeeze(sum(permute(fij(:,d,[1:(d-1),(d+1):n],d),[2,1,3,4]),2)) == -1 + fext([1:(d-1),(d+1):n],d);
            end
            for i=1:n
                fij(i,i,:,:) == 0;
                squeeze(sum(fij(i,:,[1:(i-1),(i+1):n],[1:(i-1),(i+1):n]),2)) - squeeze(sum(permute(fij(:,i,[1:(i-1),(i+1):n],[1:(i-1),(i+1):n]),[2,1,3,4]),2)) == 0;
            end
            for o=1:n
                fext(o,o) == 0;
                f(o,o) == 0;
                fij(:,o,o,:) == 0;
            end
            for d=1:n
                fij(d,:,:,d) == 0;
            end
            for o=1:n
                for d=1:n
                    if o ~= d
                        f(o,d) + fext(o,d) == 1;
                    end
                end
            end
            for o=1:n
                for d=1:n
                    prices(o,d)*demand(o,d)*f(o,d) >= sum(sum(demand(o,d).*fij(:,:,o,d).*op_link_cost./a_nom));
                    %ver eso
                end
            end
            if iter == niters
                for i=1:n
                    if s_prev(i) <= 0.1*sigma
                        s_prim(i) == 0;
                    end
                    if s_h_prev(i) <= 0.1
                        s_h_prim(i) == 0;
                    end
                    for j=1:n
                        if a_prev(i,j) <= 0.1
                            a(i,j) == 0;
                         end
                     end
                 end
             end
    
        cvx_end
        a_prev = a;
        s_prev = s_prim;
        s_h_prev = s_h_prim;
        att_dem = sum(sum(demand.*f))/sum(sum(demand));
        disp(['iter = ',num2str(iter),', beta = ',num2str(beta_or),', lam = ',num2str(lam),', nlinks =',num2str(sum(sum(a > 0.1))),', nhubs = ', ...
            num2str(sum(s_h_prim > 0.1)),', nairports = ',num2str(sum(s_prim+s_h_prim > 0.1)), ...
            ', nspokes = ',num2str(sum(s_prim > 0.1*sigma)),', att_dem = ',num2str(att_dem),', SFO = ',num2str(s_h_prim(22)),', CHI = ',num2str(s_h_prim(4))]);
        if (sum(sum(a > 0.1)) < 1)
            disp(['abandono para beta = ',num2str(beta_or),', lam = ',num2str(lam)]);
            break;
        end

        if iter == (niters-1)
            budget = get_budget(s_prim,s_h_prim,a,n,...
            station_cost,station_hub_cost,station_capacity_slope,link_cost,lam);
            
            [obj_val,pax_obj,op_obj] = get_obj_val(alfa,a, s_h_prim,s_prim,delta_s,delta_s_h,delta_s_prim,fij,f,fext,dm_pax,dm_op,coef_logit);

            filename = sprintf('./aero_results/prelim_symmetric_lastiter_logcoef03_pesimistic_sol_beta=%d_lam=%d.mat',beta_or,lam);
            save(filename,'s','s_prim','s_h','s_h_prim','delta_s_h','delta_s','delta_s_prim', ...
                'a','f','fext','fij','obj_val','pax_obj','op_obj','budget');
        end
    end
    comp_time = toc;
    disp('finished');
    
    
 
    budget =get_budget(s_prim,s_h_prim,a,n,...
    station_cost,station_hub_cost,station_capacity_slope,link_cost,lam);
    [obj_val,pax_obj,op_obj] = get_obj_val(alfa,a, s_h_prim,s_prim,delta_s,delta_s_h,delta_s_prim,fij,f,fext,dm_pax,dm_op,coef_logit);

    filename = sprintf('./aero_results/prelim_symmetric_logcoef03_pesimistic_sol_beta=%d_lam=%d.mat',beta_or,lam);
    save(filename,'s','s_prim','s_h','s_h_prim','delta_s_h','delta_s','delta_s_prim', ...
        'a','f','fext','fij','obj_val','pax_obj','op_obj','comp_time','budget');
end
function [n,link_cost,station_cost,...
    station_capacity_slope,demand,prices,...
    op_link_cost,congestion_coef_airline, congestion_coef_airport,...
    travel_time,alt_time,alt_price,a_nom,tau,sigma,...
    a_max,trans_time,s_max,station_hub_cost,distance] = parameters_aero_network()
    n = 25; 
    CAB_data = readtable('./CAB_data.xlsx');
    coor_x = table2array(CAB_data(1:25,1));
    coor_y = table2array(CAB_data(1:25,2));
    demand = table2array(CAB_data(26:50,1:25));
    demand = 1.11214.*demand.*(4/30);
 %   midwest = [5,8,9,15,21];
  %  demand(midwest,midwest) = 5*demand(midwest,midwest);
    distance = zeros(n);
    for i=1:n
        distance(i,i) = 0;
        for j=i+1:n
            distance(i,j) = sqrt((coor_x(i) - coor_x(j)).^2 + ...
            (coor_y(i)-coor_y(j)).^2);
            distance(j,i) = distance(i,j);
        end
    end
    
    %cost of using the alternative network for each o-d pair
    rng(1,"twister"); %seed
    
    Prob_nonstop=0.2;
    alt_cost = zeros(n);
    for i=1:n
        for j=i+1:n
            dist_ij=distance(i,j);
            nonstopflight = rand(5,1)<Prob_nonstop;
            distance_compet = (nonstopflight*dist_ij + (1-nonstopflight).*(720 +(0.5+0.7*rand(5,1)+0.5+0.7*rand(5,1))*dist_ij))*1e-4;
            distance_compet = (nonstopflight*dist_ij + (1-nonstopflight).*(720 +(1 + 0.2*randn(5,1).^2 + 0.2*randn(5,1).^2)*dist_ij))*1e-4;
            if (i==12) && (j == 17)
                nonstopflight = rand(5,1) < 0.8;
                distance_compet = (nonstopflight*dist_ij + (1-nonstopflight).*(720 +(1 + 0.2*randn(5,1).^2 + 0.2*randn(5,1).^2)*dist_ij))*1e-4;
            end
            alt_cost(i,j)=-log(mean(exp(-distance_compet)))*1e4;
            alt_cost(j,i) = alt_cost(i,j);
            
        end
    end
    
    %fixed cost for constructing links
    link_cost = 1e-1.*ones(n); %poner muy pequeño, si no forzar sparsity aparte.
    %digo que es el estudio de la ruta
    
    %fixed cost for constructing stations
    station_cost = 2e3.*ones(1,n); %oficinas, tasas del aeropuerto 

    station_hub_cost = 5e3.*ones(1,n);
    
    station_capacity_slope = (5*5e2 + 4*50*8).*ones(1,n); %500 e dia por estacionamiento aeronaves + 50e/hora personal*4pax

    % Congestion Coefficients
    congestion_coef_airline = 0.1;
    congestion_coef_airport = 0.1;
    
    % Prices
    
    prices = (1.7549.*(distance).^(0.7) + 0.088158.*(demand.^1.1).*mean(mean(distance.^0.7))./mean(mean(demand.^1.1)));
 %   prices = (1.058/1.1214).*prices;
 %   prices = (1.843058*0.6.*(distance).^(0.7) + 1.843058*0.4.*(demand.^1.1).*mean(mean(distance.^0.7))./mean(mean(demand.^1.1)));
    %1.843058
  %  prices = (0.9.*(distance).^(0.7) + 0.27.*(demand.^1.1).*mean(distance.^0.7)./mean(demand.^1.1));
%    prices = prices.*(72.9502/142.2529);
 %   prices = (0.9.*(distance).^(0.7) + 0.27.*(demand.^1.1).*mean(distance.^0.7)./mean(demand.^1.1));
  %  prices = prices.*(72.9502/142.2529);
    
    % Travel Time
    travel_time = distance ./ 450 + 0.6; % Time in hours
    
    % Alt Time
    alt_time = alt_cost ./ 450 + 0.6; % Time in hours

    % Op Link Cost
    op_link_cost = 5e3.*travel_time; %5k/hora, lo disminuyo para esta sim
    
    trans_time = ones(n,n,n,n);
    
    for o=1:n
        for i=1:n 
            trans_time(:,:,o,o) = 0;
            trans_time(i,i,:,:) = 0;
            trans_time(o,i,o,:) = 0;
        end
    end
    
    alt_price = (1.7549.*(alt_cost).^(0.7) +  0.088158.*(demand.^1.1).*mean(mean(alt_cost.^0.7))./mean(mean(demand.^1.1)));
   % alt_price = (1.058/1.1214).*alt_price;
   % alt_price = (1.843058*0.6.*(alt_cost).^(0.7) +  1.843058*0.4.*(demand.^1.1).*mean(mean(alt_cost.^0.7))./mean(mean(demand.^1.1)));
 %   alt_price = (0.9.*(alt_cost).^(0.7) + 0.27.*(demand.^1.1).*mean(alt_cost.^0.7)./mean(demand.^1.1));
   % alt_price = (0.9.*(alt_cost).^(0.7) + 0.27.*(demand.^1.1).*mean(alt_cost.^0.7)./mean(demand.^1.1));
   % alt_price = alt_price.*(113.8453/221.9983); %media con los  datos anteriores/media actual
    
    a_nom = 220;             
    
    tau = 0.85;
    sigma = 0.18; %sigma en formulacion, ver guille. 14,15 y 16, vuelos totales.
    a_max = 1e9;
    s_max = 1e5;

    %neutro = 1.1214
    %pesimista = 1.058
    %optimista = 1.18
end