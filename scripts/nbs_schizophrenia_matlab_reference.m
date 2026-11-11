function nbs_schizophrenia_matlab_reference(nbsdir, outfile)
%NBS_SCHIZOPHRENIA_MATLAB_REFERENCE Reference NBS v1.2 run on its own example.
%
%   nbs_schizophrenia_matlab_reference(NBSDIR, OUTFILE) runs the MATLAB NBS
%   toolbox (folder NBSDIR, containing NBSrun.m and SchizophreniaExample/)
%   with the inputs the NBS GUI pre-fills to replicate Zalesky et al. (2010):
%   t-test, contrast [-1,1] (healthy controls > schizophrenia), primary
%   threshold 3.1, 5000 permutations, alpha 0.05; once with component size
%   'Extent' and once with 'Intensity'. The result is written to OUTFILE
%   (JSON) for tests/test_nbs_schizophrenia_reference.py:
%     t_upper    edge t-statistics, upper triangle in row-major order
%                (numpy.triu_indices(74, 1))
%     extent / intensity: n_components, p (one per component), edges
%                (0-based node pairs i < j of each component)
%
%   NBSrun is called with an empty GUI handle: in NBS v1.2 the one-argument
%   form fails ('S' undefined). NBSview needs the NBS folder as the current
%   folder, so the function changes into NBSDIR and returns afterwards.
%
%   Example:
%     nbs_schizophrenia_matlab_reference('../NBS1.2', ...
%         'tests/data/nbs_schizophrenia_matlab_reference.json')

outfile = char(java.io.File(outfile).getAbsolutePath());
here = pwd;
cleanup = onCleanup(@() cd(here));
cd(nbsdir); addpath(pwd);
nbsdir = pwd;

global nbs
N = 74;
ex = fullfile(nbsdir, 'SchizophreniaExample');
out = struct();
out.source = 'MATLAB NBS v1.2 (Zalesky, Fornito & Bullmore 2010), SchizophreniaExample';
out.matlab_version = version;
out.parameters = struct('test', 't-test', 'contrast', [-1 1], 'threshold', 3.1, ...
    'n_permutations', 5000, 'alpha', 0.05, 'rng_seed', 1);

sizes = {'Extent', 'Intensity'};
for s = 1:numel(sizes)
    UI = struct();
    UI.method.ui = 'Run NBS';
    UI.test.ui = 't-test';
    UI.size.ui = sizes{s};
    UI.thresh.ui = '3.1';
    UI.perms.ui = '5000';
    UI.alpha.ui = '0.05';
    UI.contrast.ui = '[-1,1]';
    UI.design.ui = fullfile(ex, 'designMatrix.txt');
    UI.exchange.ui = '';
    UI.matrices.ui = fullfile(ex, 'matrices', 'subject01.txt');
    UI.node_coor.ui = fullfile(ex, 'COG.txt');
    UI.node_label.ui = fullfile(ex, 'nodeLabels.txt');
    rng(1);
    evalc('NBSrun(UI, struct())');   % suppress the per-permutation table
    close all;

    T = full(nbs.NBS.test_stat);
    if s == 1
        % tril column-major of a symmetric matrix = triu row-major
        out.t_upper = T(tril(true(N), -1))';
    end
    r = struct('n_components', nbs.NBS.n, 'p', nbs.NBS.pval, 'edges', {{}});
    for k = 1:nbs.NBS.n
        [i, j] = find(triu(nbs.NBS.con_mat{k} | nbs.NBS.con_mat{k}'));
        r.edges{k} = sortrows([i j] - 1);
    end
    out.(lower(sizes{s})) = r;
    fprintf('%s: %d component(s), p = %s, edges = %s\n', sizes{s}, r.n_components, ...
        mat2str(r.p, 6), mat2str(cellfun(@(e) size(e, 1), r.edges)));
end

fid = fopen(outfile, 'w');
fprintf(fid, '%s', jsonencode(out, 'PrettyPrint', false));
fclose(fid);
fprintf('Written: %s\n', outfile);
end
