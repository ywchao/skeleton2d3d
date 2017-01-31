
n_batches = 64;

out_dir = ['pbs/smplify/b' num2str(n_batches,'%02d') '/'];
makedir(out_dir);

temp_file = 'pbs/smplify/template.pbs';

pre_dir = cell(3,1);
new_str = cell(3,1);

pre_str{1} = '${exp_name}';
pre_str{2} = '${n_batches}';
pre_str{3} = '${batch_id}';

fprintf('generating smplify pbs ... \n');

for i = 1:n_batches
    exp_name = sprintf('smplify_%02d_%02d',n_batches,i);
    new_str{1} = exp_name;
    new_str{2} = num2str(n_batches);
    new_str{3} = num2str(i);
    
    flag_indent = true;
    src_file = temp_file;
    trg_file = [out_dir 'run_' num2str(i,'%02d') '.pbs'];
    if ~exist(trg_file,'file')
        C = read_file_lines(src_file,flag_indent);
        for j = 1:numel(pre_str)
            C = cellfun(@(x)strrep(x,pre_str{j},new_str{j}),C,'UniformOutput',false);
        end
        C = [C; {''}];  %#ok
        write_file_lines(trg_file,C);
    end
end

fprintf('done.\n');