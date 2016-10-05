figure(1);

% choose experiment

% exp_name = 'hg-8-lr-1e-3-bs-64-prior-0';   maxy11 = 1e6;  maxy21 = 1e6;  maxy12 = 400;  maxy22 = 1000;  clr = 'r';
% epoch_size = 572;
% exp_name = 'hg-16-lr-1e-3-bs-64-prior-0';  maxy11 = 1e6;  maxy21 = 1e6;  maxy12 = 400;  maxy22 = 1000;  clr = 'g';
% epoch_size = 572;
% exp_name = 'hg-32-lr-1e-3-bs-64-prior-0';  maxy11 = 1e6;  maxy21 = 1e6;  maxy12 = 400;  maxy22 = 1000;  clr = 'b';
% epoch_size = 572;

% exp_name = 'hg-8-lr-1e-3-bs-64-prior-10';   maxy11 = 1e6;  maxy21 = 1e6;  maxy12 = 400;  maxy22 = 1000;  clr = 'r';
% epoch_size = 572;
% exp_name = 'hg-16-lr-1e-3-bs-64-prior-10';  maxy11 = 1e6;  maxy21 = 1e6;  maxy12 = 400;  maxy22 = 1000;  clr = 'g';
% epoch_size = 572;
% exp_name = 'hg-32-lr-1e-3-bs-64-prior-10';  maxy11 = 1e6;  maxy21 = 1e6;  maxy12 = 400;  maxy22 = 1000;  clr = 'b';
% epoch_size = 572;

disp_int = 100;

% set parameters
max_epoch = 30;

format = '%s %s %s %s %s %s';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
log_file = ['./exp/h36m/' exp_name '/train.log'];
f = fopen(log_file);
C = textscan(f,format);
fclose(f);
epoch = cellfun(@(x)str2double(x),C{1}(2:end));
iter = cellfun(@(x)str2double(x),C{2}(2:end));
loss = cellfun(@(x)str2double(x),C{5}(2:end));
err = cellfun(@(x)str2double(x),C{6}(2:end));

ind = [];
for i = 1:max_epoch
    ii = find(epoch == i);
    if isempty(ii)
        continue
    end
    % sample index uniformly
    ind = [ind; ii(1:disp_int:numel(ii))];  %#ok
    % add the last iter of each epoch
    if ismember(epoch_size*i, ii) && ~ismember(epoch_size*i, ind)
        ind = [ind; epoch_size*i];  %#ok
    end
end
it = (epoch(ind)-1)*epoch_size + iter(ind);

subplot('Position',[0.05+0/2 0.55 1/2-0.05 0.4]);
plot(it,loss(ind),['-' clr]); hold on;
grid on;
axis([0 it(end) ylim]);
title('training loss');
xlabel('iteration');

subplot('Position',[0.05+1/2 0.55 1/2-0.05 0.4]);
plot(it,err(ind),['-' clr]); hold on;
grid on;
axis([0 it(end) ylim]);
title('training error');
xlabel('iteration');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
log_file = ['./exp/h36m/' exp_name '/val.log'];
f = fopen(log_file);
C = textscan(f,format);
fclose(f);
epoch = cellfun(@(x)str2double(x),C{1}(2:end));
iter = cellfun(@(x)str2double(x),C{2}(2:end));
loss = cellfun(@(x)str2double(x),C{5}(2:end));
err = cellfun(@(x)str2double(x),C{6}(2:end));

it = (epoch-1)*epoch_size + iter;
ii = mod(1:numel(it),10) == 0;

subplot('Position',[0.05+0/2 0.05 1/2-0.05 0.4]);
plot(it,loss,['-' clr]); hold on;
plot(it(ii),loss(ii),['o' clr],'MarkerSize',5);
grid on;
title('validation loss');
xlabel('iteration');

subplot('Position',[0.05+1/2 0.05 1/2-0.05 0.4]);
plot(it,err,['-' clr]); hold on;
plot(it(ii),err(ii),['o' clr],'MarkerSize',5);
grid on;
title('validation error');
xlabel('iteration');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot('Position',[0.05+0/2 0.55 1/2-0.05 0.4]); set(gca,'fontsize',6);
subplot('Position',[0.05+1/2 0.55 1/2-0.05 0.4]); set(gca,'fontsize',6);
subplot('Position',[0.05+0/2 0.05 1/2-0.05 0.4]); set(gca,'fontsize',6);
subplot('Position',[0.05+1/2 0.05 1/2-0.05 0.4]); set(gca,'fontsize',6);

subplot('Position',[0.05+0/2 0.55 1/2-0.05 0.4]);
lim = [xlim, ylim];
axis([lim(1:2) 0 maxy11]);
subplot('Position',[0.05+0/2 0.05 1/2-0.05 0.4]);
axis([lim(1:2) 0 maxy21]);

subplot('Position',[0.05+1/2 0.55 1/2-0.05 0.4]);
lim = [xlim, ylim];
axis([lim(1:2) 0 maxy12]);
subplot('Position',[0.05+1/2 0.05 1/2-0.05 0.4]);
axis([lim(1:2) 0 maxy22]);

% save to file
save_file = ['outputs/plot_' exp_name '.pdf'];
if ~exist(save_file,'file')
    set(gcf,'PaperPosition',[0 0 8 6]);
    set(gcf,'PaperOrientation','landscape');
    print(gcf,save_file,'-dpdf');
end

close;