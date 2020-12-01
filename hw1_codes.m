%% FDD3434 HW #1 CODE
%% DATASET PRE-PROCESS
dataset=importdata('zoo.data');
features=dataset.data;
labels=dataset.textdata;
% ignore types
types=features(:,end);
features=features(:,1:end-1);

% handle the numeric attribute
features(:,13)=sign(features(:,13)-4.5);
features(features<0)=0;

%convert types to color
color_types=[];
for km=1:101
    if types(km)==1
        color_types=[color_types 'r'];
    elseif types(km)==2
        color_types=[color_types 'g'];
    elseif types(km)==3
        color_types=[color_types 'b'];
    elseif types(km)==4
        color_types=[color_types 'c'];
    elseif types(km)==5
        color_types=[color_types 'm'];
    elseif types(km)==6
        color_types=[color_types 'y'];
    else
        color_types=[color_types 'k'];
    end
end
%% PCA
coeff = pca(features);
PCA_2D=features*coeff(:,1:2);

% plot colored types + names
figure
for km=1:101
    plot (PCA_2D(km,1), PCA_2D(km,2),'LineWidth',3,'Marker','*','MarkerEdgeColor',color_types(km),'MarkerSize',20);
    hold on;
end
title(['PCA results with names'],'FontSize',20);

%text(PCA_2D(:,1), PCA_2D(:,2),labels,'FontSize',16)
%% MDS
distance_matrix = pdist(features);
distance_matrix = squareform(distance_matrix);

S=-0.5*(distance_matrix-ones(101,1)*mean(distance_matrix,1)-mean(distance_matrix,2)*ones(1,101)+mean(mean(distance_matrix))*ones(101,101));

[V,D,W] = eig(S); 

[d,ind] = sort(diag(D));
Ds = D(ind,ind);
Vs = V(:,ind);

X=sqrt(Ds)*Vs';
X=X(end-1:end,:);

% plot colored types + names
figure
for km=1:101
    plot (X(1,km), X(2,km),'LineWidth',3,'Marker','*','MarkerEdgeColor',color_types(km),'MarkerSize',20);
    hold on;
end
title(['MDS results with names'],'FontSize',20);

%text(X(1,:), X(2,:),labels,'FontSize',16)
%% MODIFIED MDS
modified_features=features;
modified_features(:,2)=modified_features(:,2)*2;
modified_features(:,12)=modified_features(:,12)*2;
modified_features(:,10)=modified_features(:,10)*2;

distance_matrix = pdist(modified_features);
distance_matrix = squareform(distance_matrix);

S=-0.5*(distance_matrix-ones(101,1)*mean(distance_matrix,1)-mean(distance_matrix,2)*ones(1,101)+mean(mean(distance_matrix))*ones(101,101));

[V,D,W] = eig(S); 

[d,ind] = sort(diag(D));
Ds = D(ind,ind);
Vs = V(:,ind);

X=sqrt(Ds)*Vs';
X=X(end-1:end,:);

% plot colored types + names
figure
for km=1:101
    plot (X(1,km), X(2,km),'LineWidth',3,'Marker','*','MarkerEdgeColor',color_types(km),'MarkerSize',20);
    hold on;
end
title(['MDS results with different importance on attributes'],'FontSize',20);

%text(X(1,:), X(2,:),labels,'FontSize',16)
%% ISOMAP
distance_matrix = pdist(features);
distance_matrix = squareform(distance_matrix);

neighbour_matrix= zeros(size(distance_matrix));
number_of_neighbours=20;
for km=1:101
    [B,I] = mink(distance_matrix(km,:),number_of_neighbours);
    neighbour_matrix(km,I) = 1;
end

D = all_shortest_paths(sparse(neighbour_matrix));

S=-0.5*(D-ones(101,1)*mean(D,1)-mean(D,2)*ones(1,101)+mean(mean(D))*ones(101,101));

[V,eigen_val,W] = eig(S); 

[d,ind] = sort(diag(eigen_val));
eigen_vals = eigen_val(ind,ind);
Vs = V(:,ind);

X=sqrt(eigen_vals)*Vs';
X=X(end-1:end,:);

% plot colored types
figure
for km=1:101
    plot (X(2,km), X(1,km),'LineWidth',3,'Marker','*','MarkerEdgeColor',color_types(km),'MarkerSize',20);
    hold on;
end
title(['Isomap with p = 20'],'FontSize',20);

%text(X(2,:), X(1,:),labels,'FontSize',16)

%% SHORTEST PATH FUNCTION and other helper functions
function [D,P] = all_shortest_paths(A,varargin)

    [trans check full2sparse] = get_matlab_bgl_options(varargin{:});
    if full2sparse && ~issparse(A), A = sparse(A); end

    options = struct('algname', 'auto', 'inf', Inf, 'edge_weight', 'matrix');
    options = merge_options(options,varargin{:});

    % edge_weights is an indicator that is 1 if we are using edge_weights
    % passed on the command line or 0 if we are using the matrix.
    %edge_weights = 0;
    edge_weight_opt = 'matrix';

    if strcmp(options.edge_weight, 'matrix')
        % do nothing if we are using the matrix weights
    else
        edge_weight_opt = options.edge_weight;
    end

    if check
        % check the values of the matrix
        check_matlab_bgl(A,struct('values',1));

        % set the algname
        if strcmpi(options.algname, 'auto')
            nz = nnz(A);
            if (nz/(numel(A)+1) > .1)
                options.algname = 'floyd_warshall';
            else
                options.algname = 'johnson';
            end
        end
    else
        if strcmpi(options.algname, 'auto')
            error('all_shortest_paths:invalidParameter', ...
                'algname auto is not compatible with no check');       
        end
    end

    if trans, A = A'; end

    if nargout > 1
        [D,P] = matlab_bgl_all_sp_mex(A,lower(options.algname),options.inf,edge_weight_opt);
        P = P';
    else
        D = matlab_bgl_all_sp_mex(A,lower(options.algname),options.inf,edge_weight_opt);
    end

    if trans, D = D'; end
end

function [trans check full2sparse] = get_matlab_bgl_options(varargin)
    doptions = set_matlab_bgl_default();
    if nargin>0
        options = merge_options(doptions,varargin{:});
    else
        options = doptions;
    end

    trans = ~options.istrans;
    check = ~options.nocheck;
    full2sparse = options.full2sparse;
end

function options = merge_options(default_options,varargin)

    if ~isempty(varargin) && mod(length(varargin),2) == 0
        options = merge_structs(struct(varargin{:}),default_options);
    elseif length(varargin)==1 && isstruct(varargin{1})
        options = merge_structs(varargin{1},default_options);
    elseif ~isempty(varargin)
        error('matlag_bgl:optionsParsing',...
            'There were an odd number of key-value pairs of options specified');
    else
        options = default_options;
    end

end

function check_matlab_bgl(A,options)
    if ~isfield(options, 'nodefault') || options.nodefault == 0
        if size(A,1) ~= size(A,2)
            error('matlab_bgl:invalidParameter', 'the matrix A must be square.');
        end
    end

    if isfield(options, 'values') && options.values == 1
        if ~isa(A,'double')
            error('matlab_bgl:invalidParameter', 'the matrix A must have double values.');
        end
    end

    if isfield(options, 'noneg') && options.noneg == 1
        v=min(min(A));
        if ~isempty(v) && v < 0
            error('matlab_bgl:invalidParameter', 'the matrix A must have non-negative values.');
        end
    end

    if isfield(options, 'sym') && options.sym == 1
        if ~isequal(A,A')
            error('matlab_bgl:invalidParameter', 'the matrix A must be symmetric.');
        end
    end

    if isfield(options, 'nosparse') && options.nosparse == 1
    else
        if ~issparse(A)
            error('matlab_bgl:invalidParameter', 'the matrix A must be sparse.  (See set_matlab_bgl_default.)');
        end
    end

    if isfield(options,'nodiag') && options.nodiag == 1
        if any(diag(A))
            error('matlab_bgl:invalidParameter',...
                'the matrix A must not have any diagonal values')
        end
    end

end

function old_default = set_matlab_bgl_default(varargin)

    persistent default_options;
    if ~isa(default_options,'struct')
        % initial default options
        default_options = struct('istrans', 0, 'nocheck', 0, 'full2sparse', 0);
    end

    if nargin == 0
        old_default = default_options;
    else
        old_default = default_options;
        default_options = merge_options(default_options,varargin{:});
    end
end
