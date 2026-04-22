%% Research code by Agus Hasan

clear;
clc;

load xArraywithout1.mat;
load xArraywithout2.mat;

%% Simulation Time
tf  = 20;
dt  = 0.001;
t   = dt:dt:tf;

%% System description
m    = 23.8;
Izz  = 1.76;
xg   = 0.046;
Xud  = -2;
Yvd  = -10;
Yrd  = 0;
Nvd  = 0;
Nrd  = -1;

%% Number of states
n = 6;

%% System parameters
M    = [m-Xud 0 0;0 m-Yvd m*xg-Yrd;0 m*xg-Nvd Izz-Nrd];

A   = eye(6);
B_t = [1 0 0;0 1 0; 0 0 1];
B   = dt*[0 0 0;0 0 0;0 0 0;inv(M)*B_t];
C   = eye(6);

%% noise
QF = 0.01*eye(rank(A));
RF = 1*eye(rank(C));

%% Initialization
x        = [4;0;0;0;0;0];
xbar     = [4;0;0;0;0;0];
Pplus    = eye(rank(A));
theta    = [0 0 0]';
thetabar = [0 0 0]';
 
%% Paramater
m11 = M(1,1);
m22 = M(2,2);
m23 = M(2,3);
m32 = M(3,2);

Xu   = -0.7225;
Xuu  = -1.3274;
Yv   = -0.8612;
Yvv  = -36.2823;
Yr   = 0.1079;
Nv   = 0.1052;
Nr  = -0.5;
Nrr = -1;

%% Control input
u     = [50 10 1]';

%% Parameters for AEKF
Psi         = -B*diag(u);
S           = 0.1;
UpsilonPlus = 0*B;
lambda      = 0.9;
a           = 0.999;

%% Parameters for AO
lambdav = 0.95;
lambdat = 0.9;
Rv      = 0.001*eye(n);
Rt      = 0.001*eye(n);
Pv      = 0.001*eye(n);
Pt      = 0.001*eye(3);
Gamma1  = zeros(6,3);

%% For plotting
uArray          = [];
xArray          = [];
xbarArray       = [];
thetaArray      = [];
thetabarArray   = [];

%%
% Simulation
for i=1:(tf/dt)
    
    %u     = [50 10*cos(i*dt) 10]';
    
    Psi   = -B*diag(u);

    % intermittent fault
    if i>2000
        theta(1) = 0.424;
    end
    if i>2400
        theta(1) = 0;
    end
    if i>4800
        theta(1) = 0.321;
    end        
    if i>5100
        theta(1) = 0;
    end
    if i>8400
        theta(1) = 0.743;
    end
    if i>9800
        theta(1) = 0;
    end
    if i>14200
        theta(1) = 0.563;
    end
    if i>14500
        theta(1) = 0;
    end
    if i>16300
        theta(1) = 0.213;
    end
    if i>19500
        theta(1) = 0;
    end
    
    % transient fault
    if i>6400
        theta(2) = 0.8+(0.8/(20000-6400))*(i-20000);
    end
    if i>12000
        theta(2) = 0.8;
    end    
 
    % permanent fault
    if i>4600
        theta(3) = 0.634;
    end
    
    % without fault
    % theta    = [0 0 0]';
    
    uArray         = [uArray u];
    xArray         = [xArray x];
    thetaArray     = [thetaArray theta];
    xbarArray      = [xbarArray xbar];
    thetabarArray  = [thetabarArray thetabar]; 
    
    c13 = -m22*x(5)-((m23+m32)/2)*x(6);
    c23 = m11*x(4);
    Cv  = [0 0 c13; 0 0 c23;-c13 -c23 0];
    Dv  = -[Xu+Xuu*abs(x(4)) 0 0;0 Yv+Yvv*abs(x(5)) Yr;0 Nv Nr+Nrr*abs(x(6))];
    
    x = A*x+dt*[cos(x(3))*x(4)-sin(x(3))*x(5);sin(x(3))*x(4)+cos(x(3))*x(5);x(6);-inv(M)*(Cv+Dv)*[x(4);x(5);x(6)]]+B*u+Psi*theta+QF*dt*randn(6,1);
    y = C*x+RF*dt*randn(6,1);
    
    % Estimation using Adaptive Observer
    Kv = Pv*inv(Pv+Rv);
    Kt = Pt*Gamma1'*inv(Gamma1*Pt*Gamma1'+Rt);
    Gamma1 = (eye(n)-Kv)*Gamma1;

    xbar = xbar+(Kv+Gamma1*Kt)*(y-xbar);
    thetabar = thetabar-Kt*(y-xbar);

    c13b = -m22*xbar(5)-((m23+m32)/2)*xbar(6);
    c23b = m11*xbar(4);
    Cvb  = [0 0 c13; 0 0 c23;-c13 -c23 0];
    Dvb  = -[Xu+Xuu*abs(xbar(4)) 0 0;0 Yv+Yvv*abs(xbar(5)) Yr;0 Nv Nr+Nrr*abs(xbar(6))];    
    
    xbar      = A*xbar+dt*[cos(xbar(3))*xbar(4)-sin(xbar(3))*xbar(5);sin(xbar(3))*xbar(4)+cos(xbar(3))*xbar(5);xbar(6);-inv(M)*(Cvb+Dvb)*[xbar(4);xbar(5);xbar(6)]]+B*u+Psi*thetabar;
    thetabar = thetabar;
    Pv = (1/lambdav)*eye(n)*(eye(n)-Kv)*Pv*eye(n);
    Pt = (1/lambdat)*(eye(3)-Kt*Gamma1)*Pt;
    Gamma1 = eye(n)*Gamma1-Psi;

end

figure(1)
plot(xArraywithout1,xArraywithout2, 'g', 'LineWidth', 6)
hold on;
plot(xbarArray(1,:),xbarArray(2,:), 'b:', 'LineWidth', 6)
hold on;
plot(xArray(1,1),xArray(2,1), 'O', 'LineWidth', 20)
hold on;
plot(xArraywithout1(end),xArraywithout2(end), 'O', 'LineWidth', 20)
hold on;
plot(xbarArray(1,end),xbarArray(2,end), 'O', 'LineWidth', 20)
xlabel('$x$', 'Interpreter', 'latex','FontSize',48)
ylabel('$y$', 'Interpreter', 'latex','FontSize',48)
grid on;
grid minor;
set(gca,'FontSize',36)
legend('trajectory without fault','trajectory with fault','start','end without fault','end with fault','FontSize',48);

figure(2)
subplot(3,1,1)
plot(t,thetaArray(1,:), 'g', 'LineWidth', 6)
hold on;
plot(t,thetabarArray(1,:), 'b:', 'LineWidth', 6)
ylabel('$\theta_u^{ac}$', 'Interpreter', 'latex','FontSize',48)
grid on;
grid minor;
set(gca,'FontSize',36)
ylim([-0.05 1])
subplot(3,1,2)
plot(t,thetaArray(2,:), 'g', 'LineWidth', 6)
hold on;
plot(t,thetabarArray(2,:), 'b:', 'LineWidth', 6)
ylabel('$\theta_v^{ac}$', 'Interpreter', 'latex','FontSize',48)
grid on;
grid minor;
set(gca,'FontSize',36)
ylim([-0.05 1])
subplot(3,1,3)
plot(t,thetaArray(3,:), 'g', 'LineWidth', 6)
hold on;
plot(t,thetabarArray(3,:), 'b:', 'LineWidth', 6)
legend('true $\theta_{ac}$','estimated $\theta_{ac}$', 'Interpreter', 'latex','FontSize',48);
ylabel('$\theta_r^{ac}$', 'Interpreter', 'latex','FontSize',48)
xlabel('$t$(s)', 'Interpreter', 'latex','FontSize',48)
grid on;
grid minor;
set(gca,'FontSize',36)
ylim([-0.05 1])

figure(3);
subplot(3,1,1)
plot(t,thetaArray(1,:)-thetabarArray(1,:), ':g', 'LineWidth', 6)
grid on;
grid minor;
ylabel('Error $\theta_u^{ac}$', 'Interpreter', 'latex','FontSize',48)
set(gca,'FontSize',36)
legend('Error','FontSize',48);
ylim([-1 1])
subplot(3,1,2)
plot(t,thetaArray(2,:)-thetabarArray(2,:), ':g', 'LineWidth', 6)
grid on;
grid minor;
ylabel('Error $\theta_v^{ac}$', 'Interpreter', 'latex','FontSize',48)
set(gca,'FontSize',36)
ylim([-1 1])
subplot(3,1,3)
plot(t,thetaArray(3,:)-thetabarArray(3,:), ':g', 'LineWidth', 6)
grid on;
grid minor;
ylabel('Error $\theta_r^{ac}$', 'Interpreter', 'latex','FontSize',48)
xlabel('$t$(s)', 'Interpreter', 'latex','FontSize',48)
set(gca,'FontSize',36)
ylim([-1 1])