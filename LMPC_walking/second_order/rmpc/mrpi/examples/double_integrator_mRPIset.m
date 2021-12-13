%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Computaion of an invariant outer-epsilon approximation of the
% minimal Robust Positivly Invariant Set (mRPI),
% see Rakovic et al., Invariant Approximations of the Minimal Robust invariant Set.
% IEEE Transactions on Automatic Control 50, 3 (2005), 406�410.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% You will need the following two addons for MATLAB:
% - Multi-Parametric Toolbox 3 (MPT3):
%   http://control.ee.ethz.ch/~mpt/3/Main/Installation
% - ellipsoids - Ellipsoidal Toolbox 1.1.3 lite:
%   http://code.google.com/p/ellipsoids/downloads/list
% 

% Dynamics
A     = [1 1; 0 1];
B     = [1; 1];

% Disturbances
W  = Polyhedron([1 0;-1 0;0 1;0 -1],[1;1;1;1]);
Hw = W.A;
Kw = W.b;

%% specific choice of optimal gains in the paper
K = -[1.17 1.03];

% LQR closed-loop dynamics
A_K = A+B*K;

%pause()

% system dimension
n = W.Dim;

% initialization
alpha      = 0;       % ideally start with 0
logicalVar = 1;   
epsilon    = 5*10^-5;       % error threshold 
s          = 0;                

while logicalVar == 1

    s = s + 1;

    % inequality representation of the set W: f_i*w <= g_i , i=1,...,I_max
    f_i = (W.A)';
    g_i = W.b;
    I_max = length(W.b);

    % call of the support function h_W
    h_W   = zeros(I_max,1);
    
    for k = 1:I_max
        a = (A_K^s)' * f_i(:,k);
        h_W(k) = fkt_h_W(a, W);
    end
    
    clear('k')

    % output
    alpha_opt_s = max( h_W ./ g_i ); 
    alpha = alpha_opt_s;

    %  M(s)
    ej = eye(n);                
    sum_vec_A = zeros(n,1);
    sum_vec_B = zeros(n,1);
    updt_A    = zeros(n,1);
    updt_B    = zeros(n,1);

    for k = 1:s
        for j = 1:n
            a = (A_K^(k-1))' * ej(:,j);
            updt_A(j) = fkt_h_W(a, W);
            updt_B(j) = fkt_h_W(-a, W);
        end
        sum_vec_A = sum_vec_A + updt_A;
        sum_vec_B = sum_vec_B + updt_B;
    end
    clear('k')

    Ms = max(max(sum_vec_A, sum_vec_B));

    % Interrupt criterion
    if alpha <= epsilon/(epsilon + Ms)
        logicalVar = 0;
    end

end

% Fs
Fs = Polyhedron('A', [], 'b', [], 'Ae', eye(n), 'be', zeros(n,1));
for k = 1:s

    Fs = Fs + (A_K^(k-1)) * W;
    figure()
    plot(Fs)
    
end

% F_Inf outer-epsilon approximation
 F_alpha_s = 1/(1 - alpha) * Fs;
 figure()
 plot(F_alpha_s)
    


% support function h_W
function [h_W, diagnostics] = fkt_h_W(a, W)

% dimension of w
nn = W.Dim;

% optimization variable
w = sdpvar(nn,1);

% cost function
Objective = -a' * w;

% constraints
Constraints = W.A * w <= W.b;

% optimization
Options = sdpsettings('solver','quadprog','verbose',0);
diagnostics = optimize(Constraints, Objective, Options);

% output
w_opt = value(w);
h_W = a' * w_opt;
end
